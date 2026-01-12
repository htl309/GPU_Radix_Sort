
#include"cudasort_h.cu"
#include"core_h.cu"
#include"../Utils.hpp"


namespace GRS {

    //并行的blockdim和griddim包含了size的信息
    __global__ void floatencoder( const float* __restrict__ data,  uint32_t* __restrict__ result) {
        for (uint32_t i = 0, t = blockIdx.x * PART_SIZE + getLaneId() + 480 * (threadIdx.x >> 5); i < 15; ++i, t += LANE_COUNT) {
            uint32_t u = __float_as_uint(data[t]);
            result[t] = (u >> 31) ? ~u : (u | 0x80000000u);
        }
        __syncthreads();
    }

    __global__ void floatdecoder(  const uint32_t* __restrict__ data, float* __restrict__ result) {
        for (uint32_t i = 0, t = blockIdx.x * PART_SIZE + getLaneId() + 480 * (threadIdx.x >> 5); i < 15; ++i, t += LANE_COUNT) {
            uint32_t u = data[t];
            result[t] = __uint_as_float((u >> 31) ? (u & 0x7fffffffu) : ~u);
        }        
        __syncthreads();
    }


    //输入数组，将数组放置到256个桶中
    //将结果保存到globalHist中，它的长度是256*k_radixPasses
    //这是因为pass4回，这个全局的索引需要四个256长度的桶来放置
    __global__ void Upsweep(
        uint32_t* sort, //输入：需要排序的数组
        uint32_t* globalHist,//输出：全局的桶前缀和,长度是256*k_radixPasses
        uint32_t* passHist,//输出：这个block处理的7680个数字的桶状态,长度是7680小段的数量*256
        uint32_t sort_size,//输入：需要排序的数组的长度
        uint32_t radixShift//输入：这一次的偏移，因为有32位，每一次处理8位，所以有0 8 16 24四个档位
    ) 
    {
        //记录桶的状态，为什么用512个呢，为了效率，
        // 在后续的这个代码: s_globalHist[i] += s_globalHist[i + RADIX];将后半部分的数组加到了前半部分
        __shared__ uint32_t s_globalHist[RADIX * 2];

        //初始化一下
        for (uint32_t i = threadIdx.x; i < RADIX * 2; i += blockDim.x) {
            s_globalHist[i] = 0;
        }
        __syncthreads();
     
        {
            //我们将会开启128个线程，64 threads处理前半部分，64个线程处理后半部分
            //这样做是为了减少原子add的冲突，让程序更加的并行
            uint32_t* s_wavesHist = &s_globalHist[threadIdx.x / 64 * RADIX];

                //开始处理，将数组分割为7680的小段
            if (blockIdx.x < gridDim.x - 1) {
                const uint32_t partEnd = (blockIdx.x + 1) * VEC_PART_SIZE;
                for (uint32_t i = threadIdx.x + (blockIdx.x * VEC_PART_SIZE); i < partEnd; i += blockDim.x)
                {
                    //四个四个处理，直接将sort强制转化为uint4，这样子读取的时候一次性能读取四个一起处理
                    const uint4 t = reinterpret_cast<uint4*>(sort)[i];

                    //对应桶位置上的数字+1
                    //t.x >> radixShift & RADIX_MASK 相当于 (t.x >> radixShift)%256
                    //t.x >> radixShift处理对应的8位，%256是计算属于256的哪个进制的
                    atomicAdd(&s_wavesHist[t.x >> radixShift & RADIX_MASK], 1);
                    atomicAdd(&s_wavesHist[t.y >> radixShift & RADIX_MASK], 1);
                    atomicAdd(&s_wavesHist[t.z >> radixShift & RADIX_MASK], 1);
                    atomicAdd(&s_wavesHist[t.w >> radixShift & RADIX_MASK], 1);
                }

            }

            //最后一段数字可能不足7680个，因此这里就不四个四个处理了，直接循环的并行处理
            if (blockIdx.x == gridDim.x - 1) {
                for (uint32_t i = threadIdx.x + (blockIdx.x * PART_SIZE); i < sort_size; i += blockDim.x)
                {
                    const uint32_t t = sort[i];
                    atomicAdd(&s_wavesHist[t >> radixShift & RADIX_MASK], 1);
                }
            }
        }
        __syncthreads();

        for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
            s_globalHist[i] += s_globalHist[i + RADIX];
            //passHist长度是7680小段的数量*256
            //举个例子，如果输入的数组长度是76800，那么我们将会分成10整段进行处理
            //那么这里的passHist，长度就是10*256，也就是有10个桶
            //blockIdx.x是0-9是个数字，
            //也就是说，前十个数字都是桶号为0的值，或者说，0 10 20 .... 255*10，这256个值才属于第一个桶
            //1 11 21 .... 255*10+1 属于第二个桶
            passHist[i * gridDim.x + blockIdx.x] = s_globalHist[i];

            //现在我们对前半部分求前缀和，注意的是：这个函数是对一个wrap，也就是32个线程中的数字求前缀和
            //因为前半部分的长度是256，要是按照这样子算的话，这个数组分成了8段，
            // 我们这一步只求了每一段的前缀
            //需要注意的是，这个函数求完前缀和将数组整体往后移动了一位，最后一位变成了第一位
           //如果s_globalHist原本是1 1....1,64个1组成的数组，那么经过了这个函数之后就是32 1 2 3...31,32 1 2 3...31
            s_globalHist[i] = InclusiveWarpScanCircularShift(s_globalHist[i]);
        }
        __syncthreads();
        //因为上一步得到的s_globalHist，是分为8段的前缀和
        //我们这里需要在求一次前缀和，这里先对每一段第一个元素求前缀和
        //RADIX >> LANE_LOG=8
        if (threadIdx.x < (RADIX >> LANE_LOG)) {
            //这里又用了一个底层的trick，threadIdx.x < (RADIX >> LANE_LOG)会使__activemask=0x000000ff,前8位为1
            //需要注意的是：这里只计算0 32 64 96.... 8个位置的前缀和，并且都往后挪动一位，0位置变为0
            //继续上面的示例，如果s_globalHist是32 1 2 3...31,32 1 2 3...31，那么经过了这个函数之后就是0 1 2 3...31,32 1 2 3...31,64
            //最后的64的意思是，如果这个数组更长一些，那这个位置的数字就是原本32位置的那个数值，就是32+32=64
            s_globalHist[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_globalHist[threadIdx.x << LANE_LOG]);
        }
        __syncthreads();

        for (uint32_t i = threadIdx.x; i < RADIX; i += blockDim.x) {
            //这里的radixShift << 5就是偏移量，0 8 16 对应的就是0 256 512 的偏移量，radixShift << 5=radixShift*32
            //然后每一个warp，这里称之为Lane，都加上自己的warp第一位的数值，要是第一位就不加了，
            //为什么这里是__shfl_sync(0xfffffffe, s_globalHist[i - 1], 1)，而不是__shfl_sync(0xfffffffe, s_globalHist[i], 0)？
            //是因为0xfffffffe，0号线程跳过了，当然__shfl_sync(0xfffffffe, s_globalHist[i - 2], 2)也是可以的
            // 那么为什么不是__shfl_sync(0xfffffffe, s_globalHist[0], 1)呢？因为每组wrap的0号线程的id不一定是0，只有第一组是0
            // 之后的每组0号线程的id就是该组1号线程的id-1，1号线程的id是i，所以是i-1
            // 继续上面的示例，如果s_globalHist是0 1 2 3...31,32 1 2 3...31，那么经过了这个函数之后就是0 1 2 3...31, 32 33...63
            //还有一个地方，这里是globalHist，而s_globalHist这个只是7680个数字的桶计数
            // 这里并行的将每7680个数字中的值加到了globalHist上，这里并行值的是 blockIdx.x的并行,
            // 下面这句代码会执行RADIX*gridDim.x次
            atomicAdd(&globalHist[i + (radixShift << 5)], s_globalHist[i] + (getLaneId() ? __shfl_sync(0xfffffffe, s_globalHist[i - 1], 1) : 0));
        }
    }



    //求前缀和 
    //因为我们将一个数组分为了7680多个段去处理，因此数组中某个数字的具体位置由三部分组成：
    //globalHist+passHist+这段7680长度数字对应的桶内的位置
    //比如一个数字是200，前面199个桶中元素的数量之和放置在globalHist中
    //然后200对应的桶内位置，一部分是前面7680的桶的元素数量之和，放置在passHist中，一部分是自己的桶内元素，下一步计算
    __global__  void Scan(
        uint32_t* passHist,//输入输出:输入上一步保存的BlockDimSize*256个桶，这一步输出偏移
        uint32_t  BlockDimSize//有BlockDimSize组需要处理的7680个数字，当然最后一组可能不到7680个数字
    ) {

        //128个线程的共享内存
        __shared__ uint32_t scan[128];
        //我们会开启 256*128个线程来计算这个passHist的前缀和，256是因为我们的桶长度是256，
        //128是我们用128个线程并行的处理BlockDimSize个小段
        //如果BlockDimSize>128的话，我们就需要用循环去处理了，因为128个不够用了
        //partEnd决定了循环的次数，如果只是略大于就循环一次就行了，当然如果<128,那就不需要循环partEnd=0
        const uint32_t partEnd = BlockDimSize / blockDim.x * blockDim.x;
        //这个是位移的计算，就是整个passHist是256*BlockDimSize。0-BlockDimSize-1的位置是存放0元素桶的，我们要对其求前缀和
        //而BlockDimSize-2*BlockDimSize-1是1元素桶的，一个blockIdx.x计算一个元素
        const uint32_t offset = blockIdx.x * BlockDimSize;
        
        //因为我们要计算的前缀和是不带着自己的前缀和，比如一个长度为 32，值全为1的数组，前缀和是1 2...32
        //但是我们希望得到的是0 1 .... 31,所以要用一个circularLaneShift，将所有数字往后挪一位
        const uint32_t circularLaneShift = getLaneId() + 1 & LANE_MASK;


        uint32_t i = threadIdx.x;
        uint32_t reduction = 0;
        //开始循环计算
        for (; i < partEnd; i += blockDim.x) {
            //赋值然后求前缀和
            scan[threadIdx.x] = passHist[i+offset];
            //因为这里的threadIdx.x长度是128，所以这个被分成了4*32，4段小前缀和
            scan[threadIdx.x] = InclusiveWarpScan(scan[threadIdx.x]);
            __syncthreads();
            //第一段的前缀和是没有问题的，第二段的前缀和需要加上第一段前缀和的最后一位数字
            //第三段的前缀和需要加上第二段前缀和的最后一位数字
            //blockDim.x >> LANE_LOG=4,因为一个wrap里面是32个线程，所以分成四段
            //我们只需要用前面四个线程去做这件事情就可以
            if (threadIdx.x < (blockDim.x >> LANE_LOG)) {
                //先对32个线程的最后一个值求前缀和
                scan[(threadIdx.x + 1 << LANE_LOG) - 1] = ActiveInclusiveWarpScan(scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
            }
            __syncthreads();
            // i & ~LANE_MASK 这个是将i的后面5位设置为0.比如i在64-96之间，经过了这个i统一会变为64
            passHist[circularLaneShift + (i & ~LANE_MASK) + offset] =
                (threadIdx.x >= LANE_COUNT ?   __shfl_sync(0xffffffff, scan[threadIdx.x - 1], 0)   :   0)
                + (getLaneId() != LANE_MASK ? scan[threadIdx.x] : 0)
                +reduction;

            //(threadIdx.x >= LANE_COUNT ?   __shfl_sync(0xffffffff, scan[threadIdx.x - 1], 0)   :   0)
            //这段代码很好理解，就是前32位不用加前缀，后面开始才需要加前面的前缀
            //scan[threadIdx.x - 1]中的threadIdx.x - 1只可能是31，63和95
            
            //getLaneId() != LANE_MASK ? scan[threadIdx.x] : 0 
            //这段代码很细节，如果wrap中的ID号是31，那么就+0，为什么呢？
            //看似wrap中的ID号是31，但是由于circularLaneShift的存在，这个线程指向的其实是0位置的元素
            //比如i=127时， passHist[circularLaneShift + (i & ~LANE_MASK) + digitOffset]处理的是第96位的元素
            //因此只需要把之前95位置的元素移动到这就行了，用的就是上面的这行代码
            
            //+reduction是因为我们一次只处理128个数字，如果有更多的数字，比如有300个，
            // 那么129-128*2位的数字计算完前缀和就需要加上前面的128的最后一位，也就是 scan[blockDim.x - 1]
            //这个是一个累加的过程
            reduction += scan[blockDim.x - 1];
            __syncthreads();
        }

        //以上的操作再来一遍，用来处理剩下的数字
        if (i < BlockDimSize)
            scan[threadIdx.x] = passHist[i +  offset];
        scan[threadIdx.x] = InclusiveWarpScan(scan[threadIdx.x]);
        __syncthreads();

        if (threadIdx.x < (blockDim.x >> LANE_LOG))
        {
            scan[(threadIdx.x + 1 << LANE_LOG) - 1] =
                ActiveInclusiveWarpScan(scan[(threadIdx.x + 1 << LANE_LOG) - 1]);
        }
        __syncthreads();

        const uint32_t index = circularLaneShift + (i & ~LANE_MASK);
        if (index < BlockDimSize)
        {
            passHist[index + offset] =
                (getLaneId() != LANE_MASK ? scan[threadIdx.x] : 0) +
                (threadIdx.x >= LANE_COUNT ?
                    scan[(threadIdx.x & ~LANE_MASK) - 1] : 0) +
                reduction;
        }
    }

    //我们将数字分为BlockDimSize组，每组7680个数字
    //对于某一组的7680个数字，我们用512个线程去处理，每个线程处理15个数字
    __global__ void Sort(
        uint32_t* sort,
        uint32_t* result,//排序之后的数组
        uint32_t* globalHist,
        uint32_t* passHist,
        uint32_t sort_size,
        uint32_t radixShift
    ) {
        __shared__ uint32_t s_warpHistograms[PART_SIZE];
        __shared__ uint32_t s_localHistogram[RADIX];
        //将512个线程分为16部分，右移五位，将0-511映射到0 1 2 3 ... 15
        //左移8位，相当于乘上了256
        uint32_t* s_warpHist = &s_warpHistograms[WARP_INDEX << 8];

        //为什么是4096,是因为512个线程有16个wrap，每个wrap需要一个大小为256的桶
        //4096=256*16
        for (uint32_t i = threadIdx.x; i < 4096; i += blockDim.x) {
            s_warpHistograms[i] = 0;
        }

        //每个线程处理15个值
        uint32_t keys[15];

        if (blockIdx.x < gridDim.x - 1) {
            //因为32个线程能处理32*15个key，所以480*(threadIdx.x>>5)
            //总共处理15*32*16=7980个数字，这里的32是wrap的大小，16是256个线程包含16个wrap
            for (uint32_t i = 0, t = blockIdx.x * PART_SIZE + getLaneId() + 480 * (threadIdx.x >> 5); i < 15; ++i, t += LANE_COUNT) {
                keys[i] = sort[t];
            }
        }
        if (blockIdx.x == gridDim.x - 1) {
            for (uint32_t i = 0, t = blockIdx.x * PART_SIZE + getLaneId() + 480 * (threadIdx.x >> 5); i < 15; ++i, t += LANE_COUNT) {
                keys[i] = t < sort_size ? sort[t] : 0xffffffff;
             
            }
        }

        __syncthreads();

    
        //////////////////////////////////////////////////////
        //uint16_t offsets[15];
        //for (uint32_t i = 0; i < 15; ++i)
        //{
        //    offsets[i] = atomicAdd(&s_warpHist[keys[i] >> radixShift & RADIX_MASK], 1);
        //}
        //__syncthreads();

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // 这上下两段代码可以平替
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        uint16_t offsets[15];
        #pragma unroll
        for (uint32_t i = 0; i < 15; ++i)
        {
            unsigned warpFlags = 0xffffffff;
            #pragma unroll
            //这里就是举手，举8次手，举手状态和自己一样的数字就相同,下面链接的文章中有说明
            //https://zhuanlan.zhihu.com/p/1888596549224858422
            for (int k = 0; k < 8; ++k)
            {
                const bool t2 = keys[i] >> (k + radixShift) & 1;
                //举手，记录这次举手状态和自己一样的元素
                warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
            }
            const uint32_t bits = __popc(warpFlags & getLaneMaskLt());
            uint32_t preIncrementVal;
            if (bits == 0)
                preIncrementVal = atomicAdd((uint32_t*)&s_warpHist[keys[i] >> radixShift & RADIX_MASK], __popc(warpFlags));
            offsets[i] = __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;
        }
        __syncthreads();
        //////////////////////////////////////////////////////
        // 
        // 
        //exclusive prefix sum up the warp histograms

        //16个桶做前缀和
           //exclusive prefix sum up the warp histograms
        if (threadIdx.x < RADIX)
        {
            uint32_t reduction = s_warpHistograms[threadIdx.x];
            for (uint32_t i = threadIdx.x + RADIX; i < 4096; i += RADIX)
            {
                reduction += s_warpHistograms[i];
                s_warpHistograms[i] = reduction - s_warpHistograms[i];
            }

            //begin the exclusive prefix sum across the reductions
            s_warpHistograms[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
        }
        __syncthreads();

        if (threadIdx.x < (RADIX >> LANE_LOG))
            s_warpHistograms[threadIdx.x << LANE_LOG] = ActiveExclusiveWarpScan(s_warpHistograms[threadIdx.x << LANE_LOG]);
        __syncthreads();

        if (threadIdx.x < RADIX && getLaneId())
            s_warpHistograms[threadIdx.x] += __shfl_sync(0xfffffffe, s_warpHistograms[threadIdx.x - 1], 1);
        __syncthreads();

        //update offsets
        if (WARP_INDEX)
        {
#pragma unroll 
            for (uint32_t i = 0; i < 15; ++i)
            {
                const uint32_t t2 = keys[i] >> radixShift & RADIX_MASK;
                offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
            }
        }
        else
        {
#pragma unroll
            for (uint32_t i = 0; i < 15; ++i)
                offsets[i] += s_warpHistograms[keys[i] >> radixShift & RADIX_MASK];
        }

        //load in threadblock reductions
        if (threadIdx.x < RADIX)
        {
            //记录某个部分中桶的起始位置
            s_localHistogram[threadIdx.x] = globalHist[threadIdx.x + (radixShift << 5)] +
                passHist[threadIdx.x * gridDim.x + blockIdx.x] - s_warpHistograms[threadIdx.x];
        }
        __syncthreads();

        //scatter keys into shared memory
#pragma unroll
        for (uint32_t i = 0; i < 15; ++i)
            s_warpHistograms[offsets[i]] = keys[i];
        __syncthreads();

        //scatter runs of keys into device memory
        if (blockIdx.x < gridDim.x - 1)
        {
#pragma unroll 15
            for (uint32_t i = threadIdx.x; i < 7680; i += blockDim.x)
                result[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
        }

        if (blockIdx.x == gridDim.x - 1)
        {
            const uint32_t finalPartSize = sort_size - blockIdx.x * 7680;
            for (uint32_t i = threadIdx.x; i < finalPartSize; i += blockDim.x)
                result[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] + i] = s_warpHistograms[i];
        }
    }//Sort


    void CudaRadixSort::CudaSort(uint32_t* data, const uint32_t size)
    {
        const uint32_t blockdimsize = divRoundUp(size, PART_SIZE);

        uint32_t* sort;  
        //注意这里的sort和cudaresult的大小，这是给对最后一段未满7680长度也直接分配满了
        cudaMalloc((void**)&sort, blockdimsize * PART_SIZE *UINT32_T_SIZE);
        cudaMemcpy(sort, data, size * UINT32_T_SIZE, cudaMemcpyHostToDevice);


        uint32_t* cudaresult; 
        cudaMalloc((void**)&cudaresult, blockdimsize * PART_SIZE * UINT32_T_SIZE);
     
        uint32_t* globalHist;
        cudaMalloc((void**)&globalHist, 256 * k_radixPasses * UINT32_T_SIZE);
        cudaMemset(globalHist, 0, 256 * k_radixPasses * UINT32_T_SIZE);

        uint32_t* passHist;
        cudaMalloc((void**)&passHist, blockdimsize * 256 * UINT32_T_SIZE);
        cudaMemset(passHist, 0, blockdimsize * 256 * UINT32_T_SIZE);



        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        Upsweep << <blockdimsize, m_SweepThreads >> > (sort, globalHist, passHist, size, 0);
        Scan << <RADIX, m_SweepThreads >> > (passHist, blockdimsize);
        Sort << <blockdimsize, m_SortThreads >> > (sort, cudaresult, globalHist, passHist, size, 0);

        //注意这里的sort和cudaresult轮番的作为输入和输出
        Upsweep << <blockdimsize, m_SweepThreads >> > (cudaresult, globalHist, passHist, size, 8);
        Scan << <RADIX, m_SweepThreads >> > (passHist, blockdimsize);
        Sort << <blockdimsize, m_SortThreads >> > (cudaresult, sort, globalHist, passHist, size, 8);

        Upsweep << <blockdimsize, m_SweepThreads >> > (sort, globalHist, passHist, size, 16);
        Scan << <RADIX, m_SweepThreads >> > (passHist, blockdimsize);
        Sort << <blockdimsize, m_SortThreads >> > (sort, cudaresult, globalHist, passHist, size, 16);

        Upsweep << <blockdimsize, m_SweepThreads >> > (cudaresult, globalHist, passHist, size, 24);
        Scan << <RADIX, m_SweepThreads >> > (passHist, blockdimsize);
        Sort << <blockdimsize, m_SortThreads >> > (cudaresult, sort, globalHist, passHist, size, 24);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
     
        cudaMemcpy(data, sort, size * UINT32_T_SIZE, cudaMemcpyDeviceToHost);

        cudaFree(passHist);
        cudaFree(globalHist);
        cudaFree(cudaresult);
        cudaFree(sort);
        cudaDeviceSynchronize();
        return;
    }
    void CudaRadixSort::CudaSort(float* data, const uint32_t size)
    {
        const uint32_t blockdimsize = divRoundUp(size, PART_SIZE);



        uint32_t* sort;
        //注意这里的sort和cudaresult的大小，这是给对最后一段未满7680长度也直接分配满了
        cudaMalloc((void**)&sort, blockdimsize * PART_SIZE * FLOAT_SIZE);
        cudaMemset(sort, 0xffffffff, blockdimsize * PART_SIZE * FLOAT_SIZE);
        cudaMemcpy(sort, data, size * FLOAT_SIZE, cudaMemcpyHostToDevice);


        uint32_t* cudaresult;
        cudaMemset(cudaresult, 0xffffffff, blockdimsize * PART_SIZE * FLOAT_SIZE);
        cudaMalloc((void**)&cudaresult, blockdimsize * PART_SIZE * FLOAT_SIZE);

        uint32_t* globalHist;
        cudaMalloc((void**)&globalHist, 256 * k_radixPasses * FLOAT_SIZE);
        cudaMemset(globalHist, 0, 256 * k_radixPasses * FLOAT_SIZE);

        uint32_t* passHist;
        cudaMalloc((void**)&passHist, blockdimsize * 256 * FLOAT_SIZE);
        cudaMemset(passHist, 0, blockdimsize * 256 * FLOAT_SIZE);



        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        floatencoder << <blockdimsize, m_SortThreads >> > ((float*)sort, cudaresult);

        Upsweep << <blockdimsize, m_SweepThreads >> > (cudaresult, globalHist, passHist, size, 0);
        Scan << <RADIX, m_SweepThreads >> > (passHist, blockdimsize);
        Sort << <blockdimsize, m_SortThreads >> > (cudaresult, sort, globalHist, passHist, size, 0);
        //注意这里的sort和cudaresult轮番的作为输入和输出
        Upsweep << <blockdimsize, m_SweepThreads >> > (sort, globalHist, passHist, size, 8);
        Scan << <RADIX, m_SweepThreads >> > (passHist, blockdimsize);
        Sort << <blockdimsize, m_SortThreads >> > (sort, cudaresult, globalHist, passHist, size, 8);
        Upsweep << <blockdimsize, m_SweepThreads >> > (cudaresult, globalHist, passHist, size, 16);
        Scan << <RADIX, m_SweepThreads >> > (passHist, blockdimsize);
        Sort << <blockdimsize, m_SortThreads >> > (cudaresult, sort, globalHist, passHist, size, 16);
        Upsweep << <blockdimsize, m_SweepThreads >> > (sort, globalHist, passHist, size, 24);
        Scan << <RADIX, m_SweepThreads >> > (passHist, blockdimsize);
        Sort << <blockdimsize, m_SortThreads >> > (sort, cudaresult, globalHist, passHist, size, 24);

        floatdecoder << <blockdimsize, m_SortThreads >> > (cudaresult, (float*)sort);
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);

        cudaMemcpy(data, sort, size * FLOAT_SIZE, cudaMemcpyDeviceToHost);

        cudaFree(passHist);
        cudaFree(globalHist);
        cudaFree(cudaresult);
        cudaFree(sort);

        return;
    }
}