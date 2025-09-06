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

// ========== ���ܼ�ʱ���� ==========
// ȫ�ֱ����������Ƿ�������ϸ��ʱ
extern bool g_enableDetailedTiming;

// @brief ������ϸ��ʱģʽ
TRTAPI(void) enableDetailedTiming(bool enable);

// @brief ��ȡ��ǰ��ʱģʽ״̬
TRTAPI(bool) isDetailedTimingEnabled();

// @brief ������Ľṹ��
typedef struct tensorRT_nvinfer {
	// �����л�����
	nvinfer1::IRuntime* runtime;
	// ��������
	// ����ģ�͵�ģ�ͽṹ��ģ�Ͳ����Լ����ż���kernel���ã�
	// ���ܿ�ƽ̨�Ϳ�TensorRT�汾��ֲ
	nvinfer1::ICudaEngine* engine;
	// ������
	// �����м�ֵ��ʵ�ʽ�������Ķ���
	// ��engine�������ɴ���������󣬽��ж���������
	nvinfer1::IExecutionContext* context;
	// GPU�Դ�����/�������
	void** dataBuffer;
	// �����������飨������
	char** tensorNames;
	// ����������������
	int32_t numTensors;
	cudaStream_t stream;
} NvinferStruct;


// @brief ������onnxģ��תΪtensorrt�е�engine��ʽ�������浽����
TRTAPI(ExceptionStatus) onnxToEngine(const char* onnxFile, int memorySize);

TRTAPI(ExceptionStatus) onnxToEngineDynamicShape(const char* onnxFile, int memorySize, const char* nodeName,
	int* minShapes, int* optShapes, int* maxShapes);

// @brief ��ȡ����engineģ�ͣ�����ʼ��NvinferStruct�����仺��ռ�
TRTAPI(ExceptionStatus) nvinferInit(const char* engineFile, NvinferStruct** ptr);
TRTAPI(ExceptionStatus) nvinferInitDynamicShape(const char* engineFile, int maxBatahSize, NvinferStruct** ptr);
// @brief ͨ��ָ���ڵ����ƣ����ڴ��ϵ����ݿ������豸��
TRTAPI(ExceptionStatus) copyFloatHostToDeviceByName(NvinferStruct* ptr, const char* nodeName, float* data);
TRTAPI(ExceptionStatus) copyFloatHostToDeviceByNameZeroCopy(NvinferStruct* ptr, const char* nodeName, float* data, size_t elementCount);
// @brief ͨ��ָ���ڵ��ţ����ڴ��ϵ����ݿ������豸��
TRTAPI(ExceptionStatus) copyFloatHostToDeviceByIndex(NvinferStruct* ptr, int nodeIndex, float* data);

TRTAPI(ExceptionStatus) setBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int nbDims, int* dims);
TRTAPI(ExceptionStatus) setBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int nbDims, int* dims);

// @brief �����豸�ϵ�����
TRTAPI(ExceptionStatus) tensorRtInfer(NvinferStruct* ptr);

// @brief ͨ��ָ���ڵ����ƣ����豸�ϵ����ݿ������ڴ���
TRTAPI(ExceptionStatus) copyFloatDeviceToHostByName(NvinferStruct* ptr, const char* nodeName, float* data, size_t elementCount);

TRTAPI(ExceptionStatus) copyFloatDeviceToHostZeroCopy(NvinferStruct* ptr, const char* nodeName, float** hostData, size_t elementCount);

TRTAPI(void) cleanupAllPinnedMemoryPools();
TRTAPI(void) cleanupDeviceToHostPinnedMemoryPool();
TRTAPI(void) cleanupHostToDevicePinnedMemoryPool();

// @brief �ͷ��ض���С���豸�������̶��ڴ棨��ѡ�Ż���
TRTAPI(void) releaseDeviceToHostPinnedMemory(size_t byteSize);

// @brief ��ȡ�ڴ��״̬��Ϣ
TRTAPI(void) getDeviceToHostMemoryPoolInfo(int* poolSize, size_t* totalMemoryBytes);

// @brief ͨ��ָ���ڵ��ţ����豸�ϵ����ݿ������ڴ���
TRTAPI(ExceptionStatus) copyFloatDeviceToHostByIndex(NvinferStruct* ptr, int nodeIndex, float* data);

// @brief ɾ��������ڴ�
TRTAPI(ExceptionStatus) nvinferDelete(NvinferStruct* ptr);

// @brief ͨ���ڵ����ƻ�ȡ�󶨽ڵ����״��Ϣ
TRTAPI(ExceptionStatus) getBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int* dimLength, int* dims);

// @brief ͨ���ڵ��Ż�ȡ�󶨽ڵ����״��Ϣ
TRTAPI(ExceptionStatus) getBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int* dimLength, int* dims);

// ========== ������TensorRT 10.11���ݺ��� ==========
// @brief ��ȡ��������
TRTAPI(int32_t) getTensorCount(NvinferStruct* ptr);

// @brief ��ȡָ����������������
TRTAPI(const char*) getTensorNameByIndex(NvinferStruct* ptr, int32_t index);

// @brief ��������Ƿ�Ϊ��������
TRTAPI(bool) isTensorInput(NvinferStruct* ptr, const char* tensorName);

// @brief ��������Ƿ�Ϊ�������
TRTAPI(bool) isTensorOutput(NvinferStruct* ptr, const char* tensorName);


// ========== �������� ==========
// @brief C �ص��������壨�� C# ע���ã�
// hostData: ָ��Pinned Host�ڴ� (�ѿ�����ɵ�����)
// userData: �û������������ָ�루������C#�д�GCHandle��IntPtr��
typedef void(*CopyCompleteCallback)(void* hostData, void* userData, double elapsedMs);

TRTAPI(ExceptionStatus) copyFloatDeviceToHostAsync(
	NvinferStruct* ptr,
	const char* nodeName,
	size_t elementCount,
	CopyCompleteCallback callback,
	void* userData
);

// @brief �ͷ�ӳ��������ڴ�
TRTAPI(ExceptionStatus) freeMappedHostMemory(float* mappedPtr);
#endif // !TENSORRT_EXTERN_H


