#ifndef TENSORRT_EXTERN_H
#define TENSORRT_EXTERN_H


#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include "assert.h"
#include "cuda.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "common.h"
#include "exception_status.h"
#include "logging.h"
#include "logger.h"

// ========== 性能计时控制 ==========
// 全局变量，控制是否启用详细计时
extern bool g_enableDetailedTiming;

// @brief 启用详细计时模式
TRTAPI(void) enableDetailedTiming(bool enable);

// @brief 获取当前计时模式状态
TRTAPI(bool) isDetailedTimingEnabled();

// @brief 推理核心结构体
typedef struct tensorRT_nvinfer {
	// 反序列化引擎
	nvinfer1::IRuntime* runtime;
	// 推理引擎
	// 保存模型的模型结构、模型参数以及最优计算kernel配置；
	// 不能跨平台和跨TensorRT版本移植
	nvinfer1::ICudaEngine* engine;
	// 上下文
	// 储存中间值，实际进行推理的对象
	// 由engine创建，可创建多个对象，进行多推理任务
	nvinfer1::IExecutionContext* context;
	// GPU显存输入/输出缓冲
	void** dataBuffer;
	// 张量名称数组（新增）
	char** tensorNames;
	// 张量数量（新增）
	int32_t numTensors;
	cudaStream_t stream;
} NvinferStruct;


// @brief 将本地onnx模型转为tensorrt中的engine格式，并保存到本地
TRTAPI(ExceptionStatus) onnxToEngine(const char* onnxFile, int memorySize);

TRTAPI(ExceptionStatus) onnxToEngineDynamicShape(const char* onnxFile, int memorySize, const char* nodeName,
	int* minShapes, int* optShapes, int* maxShapes);

// @brief 读取本地engine模型，并初始化NvinferStruct，分配缓存空间
TRTAPI(ExceptionStatus) nvinferInit(const char* engineFile, NvinferStruct** ptr);
TRTAPI(ExceptionStatus) nvinferInitDynamicShape(const char* engineFile, int maxBatahSize, NvinferStruct** ptr);
// @brief 通过指定节点名称，将内存上的数据拷贝到设备上
TRTAPI(ExceptionStatus) copyFloatHostToDeviceByName(NvinferStruct* ptr, const char* nodeName, float* data);
TRTAPI(ExceptionStatus) copyFloatHostToDeviceByNameZeroCopy(NvinferStruct* ptr, const char* nodeName, float* data, size_t elementCount);
// @brief 通过指定节点编号，将内存上的数据拷贝到设备上
TRTAPI(ExceptionStatus) copyFloatHostToDeviceByIndex(NvinferStruct* ptr, int nodeIndex, float* data);

TRTAPI(ExceptionStatus) setBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int nbDims, int* dims);
TRTAPI(ExceptionStatus) setBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int nbDims, int* dims);

// @brief 推理设备上的数据
TRTAPI(ExceptionStatus) tensorRtInfer(NvinferStruct* ptr);

// @brief 通过指定节点名称，将设备上的数据拷贝到内存上
TRTAPI(ExceptionStatus) copyFloatDeviceToHostByName(NvinferStruct* ptr, const char* nodeName, float* data, size_t elementCount);

TRTAPI(ExceptionStatus) copyFloatDeviceToHostZeroCopy(NvinferStruct* ptr, const char* nodeName, float** hostData, size_t elementCount);

TRTAPI(void) cleanupAllPinnedMemoryPools();
TRTAPI(void) cleanupDeviceToHostPinnedMemoryPool();
TRTAPI(void) cleanupHostToDevicePinnedMemoryPool();

// @brief 释放特定大小的设备到主机固定内存（可选优化）
TRTAPI(void) releaseDeviceToHostPinnedMemory(size_t byteSize);

// @brief 获取内存池状态信息
TRTAPI(void) getDeviceToHostMemoryPoolInfo(int* poolSize, size_t* totalMemoryBytes);

// @brief 通过指定节点编号，将设备上的数据拷贝到内存上
TRTAPI(ExceptionStatus) copyFloatDeviceToHostByIndex(NvinferStruct* ptr, int nodeIndex, float* data);

// @brief 删除分配的内存
TRTAPI(ExceptionStatus) nvinferDelete(NvinferStruct* ptr);

// @brief 通过节点名称获取绑定节点的形状信息
TRTAPI(ExceptionStatus) getBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int* dimLength, int* dims);

// @brief 通过节点编号获取绑定节点的形状信息
TRTAPI(ExceptionStatus) getBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int* dimLength, int* dims);

// ========== 新增的TensorRT 10.11兼容函数 ==========
// @brief 获取张量数量
TRTAPI(int32_t) getTensorCount(NvinferStruct* ptr);

// @brief 获取指定索引的张量名称
TRTAPI(const char*) getTensorNameByIndex(NvinferStruct* ptr, int32_t index);

// @brief 检查张量是否为输入张量
TRTAPI(bool) isTensorInput(NvinferStruct* ptr, const char* tensorName);

// @brief 检查张量是否为输出张量
TRTAPI(bool) isTensorOutput(NvinferStruct* ptr, const char* tensorName);


// ========== 新增部分 ==========
// @brief C 回调函数定义（给 C# 注册用）
// hostData: 指向Pinned Host内存 (已拷贝完成的数据)
// userData: 用户传入的上下文指针（可以在C#中传GCHandle或IntPtr）
typedef void(*CopyCompleteCallback)(void* hostData, void* userData, double elapsedMs);

TRTAPI(ExceptionStatus) copyFloatDeviceToHostAsync(
	NvinferStruct* ptr,
	const char* nodeName,
	size_t elementCount,
	CopyCompleteCallback callback,
	void* userData
);

// @brief 释放映射的主机内存
TRTAPI(ExceptionStatus) freeMappedHostMemory(float* mappedPtr);
#endif // !TENSORRT_EXTERN_H


