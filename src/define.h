#define RADIX 256

//位运算用的，二进制下就是8个1
#define RADIX_MASK 255

//每个block处理128*15=1920个数字,因为一个数字顶4个
#define VEC_PART_SIZE		1920
//由于处理的时候封装成uint4了，一个相当于4个
//因此是1920*4=7680
#define PART_SIZE			7680


//General macros
#define LANE_COUNT          32							//Threads in a warp
#define LANE_MASK           31							//Mask of the lane count
#define LANE_LOG            5							//log2(LANE_COUNT)
#define WARP_INDEX          (threadIdx.x >> LANE_LOG)	//Warp of a thread



#define UINT32_T_SIZE 4
#define FLOAT_SIZE 4

