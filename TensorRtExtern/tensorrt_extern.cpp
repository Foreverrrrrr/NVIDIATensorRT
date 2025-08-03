#include "tensorrt_extern.h"
#include <string>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include "ErrorRecorder.h"
#include "NvInferPlugin.h"          
#include "NvInferPluginUtils.h"  
#include "common.h"
#include <unordered_map>
#include <mutex>
#include <thread>

bool g_enableDetailedTiming = false; 

// 启用/禁用详细计时模式
void enableDetailedTiming(bool enable) {
	g_enableDetailedTiming = enable;
	std::cout << "[CONFIG] 详细计时模式: " << (enable ? "启用" : "禁用") << std::endl;
}

// 获取当前计时模式状态
bool isDetailedTimingEnabled() {
	return g_enableDetailedTiming;
}
#define TIMING_LOG(message) if (g_enableDetailedTiming) { std::cout << message << std::endl; }
#define TIMING_LOG_STREAM(stream) if (g_enableDetailedTiming) { std::cout << stream << std::endl; }

// 主机到设备的零拷贝内存池
std::unordered_map<size_t, void*> g_hostToDevicePinnedMemoryPool;
std::mutex g_hostToDeviceMemoryPoolMutex;

// 设备到主机的零拷贝内存池
std::unordered_map<size_t, void*> g_deviceToHostPinnedMemoryPool;
std::mutex g_deviceToHostMemoryPoolMutex;

// 辅助函数：查找张量索引
int32_t findTensorIndex(NvinferStruct* ptr, const char* nodeName) {
	for (int32_t i = 0; i < ptr->numTensors; i++) {
		if (strcmp(ptr->tensorNames[i], nodeName) == 0) {
			return i;
		}
	}
	return -1;
}

// @brief 将本地onnx模型转为tensorrt中的engine格式，并保存到本地
// @param onnx_file_path_wchar onnx模型本地地址
// @param engine_file_path_wchar engine模型本地地址
// @param type 输出模型精度，
ExceptionStatus onnxToEngine(const char* onnxFile, int memorySize) {
	BEGIN_WRAP_TRTAPI
		std::string path(onnxFile);
	std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
	std::string modelPath = path.substr(0, iPos);//获取文件路径
	std::string modelName = path.substr(iPos, path.length() - iPos);//获取带后缀的文件名
	std::string modelName_ = modelName.substr(0, modelName.rfind("."));//获取不带后缀的文件名名
	std::string engineFile = modelPath + modelName_ + ".engine";
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
	parser->parseFromFile(onnxFile, 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1024 * 1024 * memorySize);
	config->setFlag(nvinfer1::BuilderFlag::kFP16);
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream filePtr(engineFile, std::ios::binary);
	if (!filePtr) {
		std::cerr << "could not open plan output file" << std::endl;
		return ExceptionStatus::Occurred;
	}
	nvinfer1::IHostMemory* modelStream = engine->serialize();
	filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	delete modelStream;
	delete engine;
	delete network;
	delete parser;
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
	END_WRAP_TRTAPI
}

ExceptionStatus onnxToEngineDynamicShape(const char* onnxFile, int memorySize, const char* nodeName,
	int* minShapes, int* optShapes, int* maxShapes)
{
	BEGIN_WRAP_TRTAPI
		std::string path(onnxFile);
	std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
	std::string modelPath = path.substr(0, iPos);//获取文件路径
	std::string modelName = path.substr(iPos, path.length() - iPos);//获取带后缀的文件名
	std::string modelName_ = modelName.substr(0, modelName.rfind("."));//获取不带后缀的文件名名
	std::string engineFile = modelPath + modelName_ + ".engine";
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
	// 定义网络属性
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1024 * 1024 * memorySize);
	config->setFlag(nvinfer1::BuilderFlag::kFP16);
	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(minShapes[0], minShapes[1], minShapes[2], minShapes[3]));
	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(optShapes[0], optShapes[1], optShapes[2], optShapes[3]));
	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxShapes[0], maxShapes[1], maxShapes[2], maxShapes[3]));

	config->addOptimizationProfile(profile);
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
	parser->parseFromFile(onnxFile, 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");
	// 创建推理引擎
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream filePtr(engineFile, std::ios::binary);
	if (!filePtr) {
		std::cerr << "could not open plan output file" << std::endl;
		return ExceptionStatus::Occurred;
	}
	nvinfer1::IHostMemory* modelStream = engine->serialize();
	filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	delete modelStream;
	delete engine;
	delete network;
	delete parser;
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
	END_WRAP_TRTAPI
}

ExceptionStatus nvinferInit(const char* engineFile, NvinferStruct** ptr) {
	BEGIN_WRAP_TRTAPI
		initLibNvInferPlugins(nullptr, "");
	// 以二进制方式读取问价
	std::ifstream filePtr(engineFile, std::ios::binary);
	if (!filePtr.good()) {
		std::cerr << "文件无法打开，请确定文件是否可用！" << std::endl;
		dup_last_err_msg("Model file reading error, please confirm if the file exists or if the format is correct.");
		return ExceptionStatus::Occurred;
	}

	size_t size = 0;
	filePtr.seekg(0, filePtr.end);	// 将读指针从文件末尾开始移动0个字节
	size = filePtr.tellg();	// 返回读指针的位置，此时读指针的位置就是文件的字节数
	filePtr.seekg(0, filePtr.beg);	// 将读指针从文件开头开始移动0个字节
	char* modelStream = new char[size];
	filePtr.read(modelStream, size);
	filePtr.close();
	NvinferStruct* p = new NvinferStruct();
	CHECKTRT(p->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
	CHECKTRT(p->runtime->setErrorRecorder(&gRecorder));
	CHECKTRT(p->engine = p->runtime->deserializeCudaEngine(modelStream, size));
	CHECKTRT(p->context = p->engine->createExecutionContext());
	CHECKCUDA(cudaStreamCreate(&(p->stream)));
	int32_t numIO = p->engine->getNbIOTensors();
	p->dataBuffer = new void* [numIO];
	p->tensorNames = new char* [numIO];
	p->numTensors = numIO;
	delete[] modelStream;

	for (int32_t i = 0; i < numIO; i++) {
		const char* tensorName = p->engine->getIOTensorName(i);
		nvinfer1::Dims dims = p->engine->getTensorShape(tensorName);
		nvinfer1::DataType type = p->engine->getTensorDataType(tensorName);

		size_t nameLen = strlen(tensorName) + 1;
		p->tensorNames[i] = new char[nameLen];
#ifdef _WIN32
		strcpy_s(p->tensorNames[i], nameLen, tensorName);
#else
		strcpy(p->tensorNames[i], tensorName);
#endif

		size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
		switch (type)
		{
		case nvinfer1::DataType::kINT32:
		case nvinfer1::DataType::kFLOAT: size *= 4; break;  // 明确为类型 float
		case nvinfer1::DataType::kHALF: size *= 2; break;
		case nvinfer1::DataType::kBOOL:
		case nvinfer1::DataType::kINT8:
		default:break;
		}
		CHECKCUDA(cudaMalloc(&(p->dataBuffer[i]), size));
	}
	*ptr = p;
	END_WRAP_TRTAPI
}

// @brief 读取本地engine模型，并初始化NvinferStruct，分配缓存空间
ExceptionStatus nvinferInitDynamicShape(const char* engineFile, int maxBatahSize, NvinferStruct** ptr) {
	BEGIN_WRAP_TRTAPI
		initLibNvInferPlugins(nullptr, "");
	// 以二进制方式读取问价
	std::ifstream filePtr(engineFile, std::ios::binary);
	if (!filePtr.good()) {
		std::cerr << "文件无法打开，请确定文件是否可用！" << std::endl;
		dup_last_err_msg("Model file reading error, please confirm if the file exists or if the format is correct.");
		return ExceptionStatus::Occurred;
	}

	size_t size = 0;
	filePtr.seekg(0, filePtr.end);	// 将读指针从文件末尾开始移动0个字节
	size = filePtr.tellg();	// 返回读指针的位置，此时读指针的位置就是文件的字节数
	filePtr.seekg(0, filePtr.beg);	// 将读指针从文件开头开始移动0个字节
	char* modelStream = new char[size];
	filePtr.read(modelStream, size);
	filePtr.close();
	NvinferStruct* p = new NvinferStruct();
	CHECKTRT(p->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
	CHECKTRT(p->runtime->setErrorRecorder(&gRecorder));
	CHECKTRT(p->engine = p->runtime->deserializeCudaEngine(modelStream, size));
	CHECKTRT(p->context = p->engine->createExecutionContext());
	CHECKCUDA(cudaStreamCreate(&(p->stream)));
	int32_t numIO = p->engine->getNbIOTensors();
	p->dataBuffer = new void* [numIO];
	p->tensorNames = new char* [numIO];
	p->numTensors = numIO;
	delete[] modelStream;
	for (int32_t i = 0; i < numIO; i++) {
		const char* tensorName = p->engine->getIOTensorName(i);
		nvinfer1::Dims dims = p->engine->getTensorShape(tensorName);
		nvinfer1::DataType type = p->engine->getTensorDataType(tensorName);
		size_t nameLen = strlen(tensorName) + 1;
		p->tensorNames[i] = new char[nameLen];
#ifdef _WIN32
		strcpy_s(p->tensorNames[i], nameLen, tensorName);
#else
		strcpy(p->tensorNames[i], tensorName);
#endif

		size_t size = std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
		switch (type)
		{
		case nvinfer1::DataType::kINT32:
		case nvinfer1::DataType::kFLOAT: size *= 4; break;  // 明确为类型 float
		case nvinfer1::DataType::kHALF: size *= 2; break;
		case nvinfer1::DataType::kBOOL:
		case nvinfer1::DataType::kINT8:
		default:break;
		}
		CHECKCUDA(cudaMalloc(&(p->dataBuffer[i]), size * maxBatahSize));
	}
	*ptr = p;
	END_WRAP_TRTAPI
}

ExceptionStatus copyFloatHostToDeviceByName(NvinferStruct* ptr, const char* nodeName, float* data)
{
	BEGIN_WRAP_TRTAPI
		int32_t tensorIndex = findTensorIndex(ptr, nodeName);
	if (tensorIndex < 0) {
		return ExceptionStatus::OccurredTRT;
	}

	// 获取张量形状信息
	nvinfer1::Dims dims = ptr->context->getTensorShape(nodeName);
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpyAsync(ptr->dataBuffer[tensorIndex], data, size * sizeof(float), cudaMemcpyHostToDevice, ptr->stream));
	END_WRAP_TRTAPI
}

ExceptionStatus copyFloatHostToDeviceByNameZeroCopy(
	NvinferStruct* ptr,
	const char* nodeName,
	float* data,
	size_t elementCount)
{
	BEGIN_WRAP_TRTAPI
		try {
		int32_t tensorIndex = findTensorIndex(ptr, nodeName);
		if (tensorIndex < 0) {
			return ExceptionStatus::OccurredTRT;
		}

		size_t byteSize = elementCount * sizeof(float);
		void* pinnedMem = nullptr;
		{
			std::lock_guard<std::mutex> lock(g_hostToDeviceMemoryPoolMutex);
			auto it = g_hostToDevicePinnedMemoryPool.find(byteSize);
			if (it != g_hostToDevicePinnedMemoryPool.end()) {
				pinnedMem = it->second;
			}
		}
		if (!pinnedMem) {
			CHECKCUDA(cudaHostAlloc(&pinnedMem, byteSize, cudaHostAllocMapped));
			std::lock_guard<std::mutex> lock(g_hostToDeviceMemoryPoolMutex);
			g_hostToDevicePinnedMemoryPool[byteSize] = pinnedMem;
		}
		std::memcpy(pinnedMem, data, byteSize);
		void* devicePtr = ptr->dataBuffer[tensorIndex];
#if __CUDA_ARCH__ >= 700
		__pipeline_memcpy_async(devicePtr, pinnedMem, byteSize);
		__pipeline_commit();
		__pipeline_wait_prior(0);
#else
		CHECKCUDA(cudaMemcpyAsync(devicePtr, pinnedMem, byteSize,
			cudaMemcpyHostToDevice, ptr->stream));
		CHECKCUDA(cudaStreamSynchronize(ptr->stream)); // 确保数据已传输到设备
#endif

	}
	catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return ExceptionStatus::Occurred;
	}
	END_WRAP_TRTAPI
}


ExceptionStatus copyFloatHostToDeviceByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	BEGIN_WRAP_TRTAPI
		if (nodeIndex < 0 || nodeIndex >= ptr->numTensors) {
			return ExceptionStatus::OccurredTRT;
		}

	const char* tensorName = ptr->tensorNames[nodeIndex];
	// 获取张量形状信息
	nvinfer1::Dims dims = ptr->context->getTensorShape(tensorName);
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpyAsync(ptr->dataBuffer[nodeIndex], data, size * sizeof(float), cudaMemcpyHostToDevice, ptr->stream));
	END_WRAP_TRTAPI
}
ExceptionStatus setBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int nbDims, int* dims)
{
	BEGIN_WRAP_TRTAPI
		switch (nbDims)
		{
		case 2:
			CHECKTRT(ptr->context->setInputShape(nodeName, nvinfer1::Dims2(dims[0], dims[1])));
			break;
		case 3:
			CHECKTRT(ptr->context->setInputShape(nodeName, nvinfer1::Dims3(dims[0], dims[1], dims[2])));
			break;
		case 4:
			CHECKTRT(ptr->context->setInputShape(nodeName, nvinfer1::Dims4(dims[0], dims[1], dims[2], dims[3])));
			break;
		default:break;
		}
	END_WRAP_TRTAPI
}
ExceptionStatus setBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int nbDims, int* dims)
{
	BEGIN_WRAP_TRTAPI
		if (nodeIndex < 0 || nodeIndex >= ptr->numTensors) {
			return ExceptionStatus::OccurredTRT;
		}

	const char* tensorName = ptr->tensorNames[nodeIndex];
	switch (nbDims)
	{
	case 2:
		CHECKTRT(ptr->context->setInputShape(tensorName, nvinfer1::Dims2(dims[0], dims[1])));
		break;
	case 3:
		CHECKTRT(ptr->context->setInputShape(tensorName, nvinfer1::Dims3(dims[0], dims[1], dims[2])));
		break;
	case 4:
		CHECKTRT(ptr->context->setInputShape(tensorName, nvinfer1::Dims4(dims[0], dims[1], dims[2], dims[3])));
		break;
	default:break;
	}
	END_WRAP_TRTAPI
}

ExceptionStatus tensorRtInfer(NvinferStruct* ptr)
{
	BEGIN_WRAP_TRTAPI
		if (!ptr->context->allInputDimensionsSpecified()) {
			std::cerr << "Not all input dimensions are specified!" << std::endl;
			return ExceptionStatus::OccurredTRT;
		}
	for (int32_t i = 0; i < ptr->numTensors; i++) {
		const char* tensorName = ptr->tensorNames[i];
		CHECKTRT(ptr->context->setTensorAddress(tensorName, ptr->dataBuffer[i]));
	}

	CHECKTRT(ptr->context->enqueueV3(ptr->stream));
	END_WRAP_TRTAPI
}

/// <summary>
/// 推理指针GPU取流到CPU
/// </summary>
/// <param name="ptr"></param>
/// <param name="nodeName"></param>
/// <param name="data"></param>
/// <returns></returns>
ExceptionStatus copyFloatDeviceToHostByName(
	NvinferStruct* ptr,
	const char* nodeName,
	float* data,
	size_t elementCount)
{
	BEGIN_WRAP_TRTAPI
		try {
		int32_t tensorIndex = findTensorIndex(ptr, nodeName);
		if (tensorIndex < 0) {
			return ExceptionStatus::OccurredTRT;
		}

		size_t byteSize = elementCount * sizeof(float);
		CHECKCUDA(cudaMemcpyAsync(
			data,
			ptr->dataBuffer[tensorIndex],
			byteSize,
			cudaMemcpyDeviceToHost,
			ptr->stream));
		CHECKCUDA(cudaStreamSynchronize(ptr->stream));
	}
	catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return ExceptionStatus::Occurred;
	}
	END_WRAP_TRTAPI
}

// 修正后的C++优化版本 - 使用正确的ExceptionStatus返回值
ExceptionStatus WaitForInferenceCompletion(NvinferStruct* ptr, int timeoutMs = 1000)
{
	if (!ptr || !ptr->stream) {
		return ExceptionStatus::NotOccurred;
	}

	auto start = std::chrono::high_resolution_clock::now();

	while (true) {
		cudaError_t streamStatus = cudaStreamQuery(ptr->stream);
		if (streamStatus == cudaSuccess) {
			TIMING_LOG("[DEBUG] 推理完成确认")
			return ExceptionStatus::NotOccurred;
		}
		else if (streamStatus != cudaErrorNotReady) {
			TIMING_LOG("[ERROR] 流查询错误: " << cudaGetErrorString(streamStatus));
			return ExceptionStatus::OccurredCuda;
		}
		auto current = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration<double, std::milli>(current - start).count();
		if (elapsed > timeoutMs) {
			TIMING_LOG("[WARNING] 等待推理完成超时: " << elapsed << "ms");
			return ExceptionStatus::NotOccurred; // 超时但继续执行
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}

// 获取最后的异常状态（供C#端调用）
ExceptionStatus GetLastExceptionStatus(NvinferStruct* ptr)
{
	// 检查CUDA错误
	cudaError_t cudaErr = cudaGetLastError();
	if (cudaErr != cudaSuccess) {
		TIMING_LOG("[DEBUG] 检测到CUDA错误: " << cudaGetErrorString(cudaErr));
		return ExceptionStatus::OccurredCuda;
	}
	return ExceptionStatus::NotOccurred;
}

// 修正后的优化传输方法
ExceptionStatus copyFloatDeviceToHostZeroCopy(
	NvinferStruct* ptr,
	const char* nodeName,
	float** hostData,
	size_t elementCount)
{
	try {
		auto func_start = std::chrono::high_resolution_clock::now();
		TIMING_LOG_STREAM("[TIMING] copyFloatDeviceToHostZeroCopy，数据大小: "
			<< (elementCount * sizeof(float) / 1024) << "KB");
		auto step1_start = std::chrono::high_resolution_clock::now();
		TIMING_LOG("[TIMING] 开始等待推理完成...");

		ExceptionStatus waitResult = WaitForInferenceCompletion(ptr, 2000);
		if (waitResult != ExceptionStatus::NotOccurred) {
			std::cout << "[ERROR] 等待推理完成失败，状态码: " << static_cast<int>(waitResult) << std::endl;
			return waitResult;
		}

		auto step1_end = std::chrono::high_resolution_clock::now();
		auto step1_time = std::chrono::duration<double, std::milli>(step1_end - step1_start).count();
		TIMING_LOG_STREAM("[TIMING] 推理等待耗时 " << step1_time << "ms");
		auto step2_start = std::chrono::high_resolution_clock::now();
		TIMING_LOG("[TIMING] 查找张量索引...");

		int32_t tensorIndex = findTensorIndex(ptr, nodeName);
		if (tensorIndex < 0) {
			std::cerr << "[ERROR] 无效节点名: " << nodeName << std::endl;
			return ExceptionStatus::OccurredTRT;
		}

		size_t byteSize = elementCount * sizeof(float);
		auto step2_end = std::chrono::high_resolution_clock::now();
		auto step2_time = std::chrono::duration<double, std::milli>(step2_end - step2_start).count();
		TIMING_LOG_STREAM("[TIMING] 张量查找耗时 " << step2_time << "ms (索引=" << tensorIndex << ")");
		auto step3_start = std::chrono::high_resolution_clock::now();
		TIMING_LOG("[TIMING] 内存池查找/分配...");
		void* pinnedMem = nullptr;
		bool memoryReused = false;
		auto pool_lookup_start = std::chrono::high_resolution_clock::now();
		{
			std::lock_guard<std::mutex> lock(g_deviceToHostMemoryPoolMutex);
			auto it = g_deviceToHostPinnedMemoryPool.find(byteSize);
			if (it != g_deviceToHostPinnedMemoryPool.end()) {
				pinnedMem = it->second;
				memoryReused = true;
				TIMING_LOG_STREAM("[TIMING] 内存池复用成功，大小: " << (byteSize / 1024) << "KB");
			}
		}
		auto pool_lookup_end = std::chrono::high_resolution_clock::now();
		auto pool_lookup_time = std::chrono::duration<double, std::milli>(pool_lookup_end - pool_lookup_start).count();

		auto alloc_start = std::chrono::high_resolution_clock::now();
		if (!pinnedMem) {
			TIMING_LOG("[TIMING] 内存池中无匹配项，开始分配新内存...");
			cudaError_t err = cudaHostAlloc(&pinnedMem, byteSize, cudaHostAllocMapped);
			if (err != cudaSuccess) {
				std::cerr << "[ERROR] cudaHostAlloc失败: " << cudaGetErrorString(err) << std::endl;
				return ExceptionStatus::OccurredCuda;
			}
			std::lock_guard<std::mutex> lock(g_deviceToHostMemoryPoolMutex);
			g_deviceToHostPinnedMemoryPool[byteSize] = pinnedMem;
			TIMING_LOG_STREAM("[TIMING] 新内存已加入池中，大小: " << (byteSize / 1024) << "KB");
		}
		auto alloc_end = std::chrono::high_resolution_clock::now();
		auto alloc_time = std::chrono::duration<double, std::milli>(alloc_end - alloc_start).count();

		auto step3_end = std::chrono::high_resolution_clock::now();
		auto step3_time = std::chrono::duration<double, std::milli>(step3_end - step3_start).count();
		TIMING_LOG_STREAM("[TIMING] 内存池操作耗时 " << step3_time
			<< "ms (查找=" << pool_lookup_time << "ms, 分配=" << alloc_time << "ms, 复用="
			<< (memoryReused ? "是" : "否") << ")");

		auto step4_start = std::chrono::high_resolution_clock::now();
		TIMING_LOG("[TIMING] 开始GPU->主机数据传输...");

		auto memcpy_start = std::chrono::high_resolution_clock::now();
		cudaError_t err = cudaMemcpyAsync(pinnedMem, ptr->dataBuffer[tensorIndex], byteSize,
			cudaMemcpyDeviceToHost, ptr->stream);
		if (err != cudaSuccess) {
			std::cerr << "[ERROR] cudaMemcpyAsync失败: " << cudaGetErrorString(err) << std::endl;
			return ExceptionStatus::OccurredCuda;
		}
		auto memcpy_launch_end = std::chrono::high_resolution_clock::now();
		auto memcpy_launch_time = std::chrono::duration<double, std::milli>(memcpy_launch_end - memcpy_start).count();

		auto sync_start = std::chrono::high_resolution_clock::now();
		err = cudaStreamSynchronize(ptr->stream);
		if (err != cudaSuccess) {
			std::cerr << "[ERROR] cudaStreamSynchronize失败: " << cudaGetErrorString(err) << std::endl;
			return ExceptionStatus::OccurredCuda;
		}
		auto sync_end = std::chrono::high_resolution_clock::now();
		auto sync_time = std::chrono::duration<double, std::milli>(sync_end - sync_start).count();

		auto step4_end = std::chrono::high_resolution_clock::now();
		auto step4_time = std::chrono::duration<double, std::milli>(step4_end - step4_start).count();
		TIMING_LOG_STREAM("[TIMING] GPU传输耗时 " << step4_time
			<< "ms (启动=" << memcpy_launch_time << "ms, 同步=" << sync_time << "ms)");
		auto func_end = std::chrono::high_resolution_clock::now();
		auto func_total_time = std::chrono::duration<double, std::milli>(func_end - func_start).count();

		*hostData = static_cast<float*>(pinnedMem);
		double transferSpeedMBps = (byteSize / 1024.0 / 1024.0) / (step4_time / 1000.0);
		double overallSpeedMBps = (byteSize / 1024.0 / 1024.0) / (func_total_time / 1000.0);

		if (g_enableDetailedTiming) {
			std::cout << "[TIMING] 推理等待: " << step1_time << "ms (" << (step1_time / func_total_time * 100) << "%)" << std::endl;
			std::cout << "[TIMING] 张量查找: " << step2_time << "ms (" << (step2_time / func_total_time * 100) << "%)" << std::endl;
			std::cout << "[TIMING] 内存池操作: " << step3_time << "ms (" << (step3_time / func_total_time * 100) << "%)" << std::endl;
			std::cout << "[TIMING] GPU传输: " << step4_time << "ms (" << (step4_time / func_total_time * 100) << "%)" << std::endl;
			std::cout << "[TIMING] 总耗时: " << func_total_time << "ms" << std::endl;
			std::cout << "[TIMING] 纯传输速度: " << transferSpeedMBps << " MB/s" << std::endl;
			std::cout << "[TIMING] 整体速度: " << overallSpeedMBps << " MB/s" << std::endl;
		}
		else {
			std::cout << "[PERF] copyFloatDeviceToHostZeroCopy: " << func_total_time
				<< "ms, " << overallSpeedMBps << "MB/s" << std::endl;
		}

		return ExceptionStatus::NotOccurred;
	}
	catch (const std::exception& e) {
		std::cerr << "[EXCEPTION] C++异常: " << e.what() << std::endl;
		return ExceptionStatus::Occurred;
	}
	catch (...) {
		std::cerr << "[EXCEPTION] 未知C++异常" << std::endl;
		return ExceptionStatus::Occurred;
	}
}

// 检查推理是否完成的简单版本（供C#调用）
bool IsInferenceComplete(NvinferStruct* ptr)
{
	if (!ptr || !ptr->stream) {
		return true;
	}

	cudaError_t streamStatus = cudaStreamQuery(ptr->stream);
	bool isComplete = (streamStatus == cudaSuccess);

	std::cout << "[DEBUG] 推理状态查询: " << (isComplete ? "完成" : "进行中") << std::endl;
	return isComplete;
}

typedef void(*CopyCompleteCallback)(void* hostData, void* userData, double elapsedMs);

ExceptionStatus copyFloatDeviceToHostAsync(
	NvinferStruct* ptr,
	const char* nodeName,
	size_t elementCount,
	CopyCompleteCallback callback,
	void* userData)
{
	BEGIN_WRAP_TRTAPI
		try {
		auto funcStart = std::chrono::high_resolution_clock::now();
		TIMING_LOG_STREAM("[TIMING-ASYNC] 开始异步数据传输，节点: " << nodeName
			<< ", 元素数: " << elementCount << ", 数据大小: " << (elementCount * sizeof(float) / 1024) << "KB");

		auto step1_start = std::chrono::high_resolution_clock::now();
		TIMING_LOG("[TIMING-ASYNC] 查找张量索引...");

		int32_t tensorIndex = findTensorIndex(ptr, nodeName);
		if (tensorIndex < 0) {
			std::cout << "[ERROR-ASYNC] 无效节点名: " << nodeName << std::endl;
			return ExceptionStatus::OccurredTRT;
		}

		size_t byteSize = elementCount * sizeof(float);
		auto step1_end = std::chrono::high_resolution_clock::now();
		auto step1_time = std::chrono::duration<double, std::milli>(step1_end - step1_start).count();
		TIMING_LOG_STREAM("[TIMING-ASYNC] 张量查找耗时 " << step1_time << "ms (索引=" << tensorIndex << ")");
		auto step2_start = std::chrono::high_resolution_clock::now();
		TIMING_LOG("[TIMING-ASYNC] 双缓冲池查找/分配...");

		struct BufferPair { void* hostBuf[2] = { nullptr, nullptr }; int index = 0; };
		static std::unordered_map<size_t, BufferPair> g_doubleBufferPool;
		static std::mutex g_mutex;
		static size_t totalBufferAllocated = 0;
		BufferPair* buffers = nullptr;
		bool newBufferCreated = false;
		auto pool_lookup_start = std::chrono::high_resolution_clock::now();
		{
			std::lock_guard<std::mutex> lock(g_mutex);
			auto it = g_doubleBufferPool.find(byteSize);
			if (it == g_doubleBufferPool.end()) {
				TIMING_LOG_STREAM("[TIMING-ASYNC] 双缓冲池中无匹配项，创建新的双缓冲对，大小: " << (byteSize / 1024) << "KB");
				BufferPair pair;
				auto allocStart = std::chrono::high_resolution_clock::now();
				CHECKCUDA(cudaMallocHost(&pair.hostBuf[0], byteSize));
				CHECKCUDA(cudaMallocHost(&pair.hostBuf[1], byteSize));
				auto allocEnd = std::chrono::high_resolution_clock::now();
				pair.index = 0;
				it = g_doubleBufferPool.emplace(byteSize, std::move(pair)).first;
				newBufferCreated = true;
				totalBufferAllocated += byteSize * 2;

				auto allocTime = std::chrono::duration<double, std::milli>(allocEnd - allocStart).count();
				TIMING_LOG_STREAM("[TIMING-ASYNC] 双缓冲分配完成，耗时: " << allocTime << "ms, "
					<< "池中总缓冲: " << g_doubleBufferPool.size() << "个, "
					<< "总内存: " << (totalBufferAllocated / 1024 / 1024) << "MB");
			}
			else {
				TIMING_LOG("[TIMING-ASYNC] 复用现有双缓冲对");
			}
			buffers = &it->second;
		}
		auto pool_lookup_end = std::chrono::high_resolution_clock::now();
		auto pool_lookup_time = std::chrono::duration<double, std::milli>(pool_lookup_end - pool_lookup_start).count();

		auto step2_end = std::chrono::high_resolution_clock::now();
		auto step2_time = std::chrono::duration<double, std::milli>(step2_end - step2_start).count();
		TIMING_LOG_STREAM("[TIMING-ASYNC] 完成: 双缓冲池操作耗时 " << step2_time << "ms (查找/分配="
			<< pool_lookup_time << "ms, 新建=" << (newBufferCreated ? "是" : "否") << ")");
		auto step3_start = std::chrono::high_resolution_clock::now();
		TIMING_LOG("[TIMING-ASYNC] 检查GPU流状态...");

		cudaError_t streamStatus = cudaStreamQuery(ptr->stream);
		bool streamBusy = (streamStatus == cudaErrorNotReady);

		auto step3_end = std::chrono::high_resolution_clock::now();
		auto step3_time = std::chrono::duration<double, std::milli>(step3_end - step3_start).count();
		TIMING_LOG_STREAM("[TIMING-ASYNC] GPU流状态检查耗时 " << step3_time << "ms (状态=" <<
			(streamStatus == cudaSuccess ? "空闲" :
				streamStatus == cudaErrorNotReady ? "忙碌" :
				cudaGetErrorString(streamStatus)) << ")");
		auto step4_start = std::chrono::high_resolution_clock::now();
		TIMING_LOG("[TIMING-ASYNC] 启动异步GPU->主机传输...");

		void* devicePtr = ptr->dataBuffer[tensorIndex];
		int writeIndex = buffers->index;

		TIMING_LOG_STREAM("[TIMING-ASYNC] 使用缓冲区索引: " << writeIndex
			<< ", 设备指针: " << devicePtr
			<< ", 主机缓冲: " << buffers->hostBuf[writeIndex]);
		auto memcpy_launch_start = std::chrono::high_resolution_clock::now();
		CHECKCUDA(cudaMemcpyAsync(
			buffers->hostBuf[writeIndex],
			devicePtr,
			byteSize,
			cudaMemcpyDeviceToHost,
			ptr->stream));
		auto memcpy_launch_end = std::chrono::high_resolution_clock::now();
		auto memcpy_launch_time = std::chrono::duration<double, std::milli>(memcpy_launch_end - memcpy_launch_start).count();

		auto step4_end = std::chrono::high_resolution_clock::now();
		auto step4_time = std::chrono::duration<double, std::milli>(step4_end - step4_start).count();
		TIMING_LOG_STREAM("[TIMING-ASYNC] 异步传输启动耗时 " << step4_time
			<< "ms (纯启动=" << memcpy_launch_time << "ms)");
		auto step5_start = std::chrono::high_resolution_clock::now();
		TIMING_LOG("[TIMING-ASYNC] 设置异步回调...");
		struct CallbackData {
			CopyCompleteCallback cb;
			void* buf;
			void* userData;
			std::chrono::high_resolution_clock::time_point startTime;
			std::chrono::high_resolution_clock::time_point funcStartTime;
			size_t byteSize;
			int bufferIndex;
			bool enableDetailedTiming;
		};

		auto* data = new CallbackData{
			callback,
			buffers->hostBuf[writeIndex],
			userData,
			memcpy_launch_start,
			funcStart,
			byteSize,
			writeIndex,
			g_enableDetailedTiming
		};
		CHECKCUDA(cudaLaunchHostFunc(ptr->stream, [](void* userData) {
			auto* d = reinterpret_cast<CallbackData*>(userData);
			auto endTime = std::chrono::high_resolution_clock::now();
			auto transferElapsed = std::chrono::duration<double, std::milli>(endTime - d->startTime).count();
			auto totalElapsed = std::chrono::duration<double, std::milli>(endTime - d->funcStartTime).count();
			double transferSpeedMBps = (d->byteSize / 1024.0 / 1024.0) / (transferElapsed / 1000.0);
			double overallSpeedMBps = (d->byteSize / 1024.0 / 1024.0) / (totalElapsed / 1000.0);

			if (d->enableDetailedTiming) {
				std::cout << "[TIMING-ASYNC-CALLBACK] 纯GPU传输时间: " << transferElapsed << "ms" << std::endl;
				std::cout << "[TIMING-ASYNC-CALLBACK] 总体异步耗时: " << totalElapsed << "ms" << std::endl;
				std::cout << "[TIMING-ASYNC-CALLBACK] 纯传输速度: " << transferSpeedMBps << " MB/s" << std::endl;
				std::cout << "[TIMING-ASYNC-CALLBACK] 整体异步速度: " << overallSpeedMBps << " MB/s" << std::endl;
				std::cout << "[TIMING-ASYNC-CALLBACK] 缓冲区索引: " << d->bufferIndex << std::endl;
				std::cout << "[TIMING-ASYNC-CALLBACK] 数据大小: " << (d->byteSize / 1024) << "KB" << std::endl;
			}
			else {
				std::cout << "[PERF-ASYNC-CALLBACK] 异步传输完成: 传输=" << transferElapsed
					<< "ms, 总计=" << totalElapsed << "ms, 速度=" << transferSpeedMBps << "MB/s" << std::endl;
			}

			if (d->cb) {
				TIMING_LOG("[TIMING-ASYNC-CALLBACK] 执行用户回调函数");
				d->cb(d->buf, d->userData, transferElapsed);
			}

			delete d;
			}, data));

		auto step5_end = std::chrono::high_resolution_clock::now();
		auto step5_time = std::chrono::duration<double, std::milli>(step5_end - step5_start).count();
		TIMING_LOG_STREAM("[TIMING-ASYNC] 回调设置耗时 " << step5_time << "ms");
		auto step6_start = std::chrono::high_resolution_clock::now();
		TIMING_LOG("[TIMING-ASYNC] 缓冲区索引切换...");
		int oldIndex = buffers->index;
		buffers->index ^= 1;
		auto step6_end = std::chrono::high_resolution_clock::now();
		auto step6_time = std::chrono::duration<double, std::milli>(step6_end - step6_start).count();
		TIMING_LOG_STREAM("[TIMING-ASYNC] 缓冲区切换耗时 " << step6_time
			<< "ms (从索引" << oldIndex << "切换到" << buffers->index << ")");
		auto funcEnd = std::chrono::high_resolution_clock::now();
		auto funcTime = std::chrono::duration<double, std::milli>(funcEnd - funcStart).count();

		if (g_enableDetailedTiming) {
			std::cout << "[TIMING-ASYNC] 张量查找: " << step1_time << "ms (" << (step1_time / funcTime * 100) << "%)" << std::endl;
			std::cout << "[TIMING-ASYNC] 双缓冲池: " << step2_time << "ms (" << (step2_time / funcTime * 100) << "%)" << std::endl;
			std::cout << "[TIMING-ASYNC] 流状态检查: " << step3_time << "ms (" << (step3_time / funcTime * 100) << "%)" << std::endl;
			std::cout << "[TIMING-ASYNC] 传输启动: " << step4_time << "ms (" << (step4_time / funcTime * 100) << "%)" << std::endl;
			std::cout << "[TIMING-ASYNC] 回调设置: " << step5_time << "ms (" << (step5_time / funcTime * 100) << "%)" << std::endl;
			std::cout << "[TIMING-ASYNC] 缓冲切换: " << step6_time << "ms (" << (step6_time / funcTime * 100) << "%)" << std::endl;
			std::cout << "[TIMING-ASYNC] 执行总耗时: " << funcTime << "ms" << std::endl;
			if (step2_time > funcTime * 0.4) {
				std::cout << "[ANALYSIS-ASYNC] 双缓冲池操作占比过高(" << (step2_time / funcTime * 100) << "%)" << std::endl;
			}
			if (newBufferCreated) {
				std::cout << "[ANALYSIS-ASYNC] 创建双缓冲池" << std::endl;
			}
			if (streamBusy) {
				std::cout << "[ANALYSIS-ASYNC] GPU流忙碌状态，异步传输将在GPU完成当前任务后执行" << std::endl;
			}
			else {
				std::cout << "[ANALYSIS-ASYNC] GPU流空闲状态，异步传输将立即开始" << std::endl;
			}
			std::cout << "[ANALYSIS-ASYNC] 异步优势: 主线程耗时仅" << funcTime << "ms，实际传输在后台进行" << std::endl;
		}
		else {
			std::cout << "[PERF-ASYNC] copyFloatDeviceToHostAsync启动: " << funcTime
				<< "ms (异步传输进行中)" << std::endl;
		}

		TIMING_LOG("[TIMING-ASYNC] 异步传输已启动，等待GPU完成...");

		return ExceptionStatus::NotOccurred;
	}
	catch (const std::exception& e) {
		std::cerr << "[EXCEPTION-ASYNC] C++异常: " << e.what() << std::endl;
		return ExceptionStatus::Occurred;
	}
	END_WRAP_TRTAPI
}


// 清理主机到设备的内存池
void cleanupHostToDevicePinnedMemoryPool()
{
	std::lock_guard<std::mutex> lock(g_hostToDeviceMemoryPoolMutex);
	for (auto& entry : g_hostToDevicePinnedMemoryPool) {
		cudaFreeHost(entry.second);
	}
	g_hostToDevicePinnedMemoryPool.clear();
}

// 清理设备到主机的内存池
void cleanupDeviceToHostPinnedMemoryPool()
{
	std::lock_guard<std::mutex> lock(g_deviceToHostMemoryPoolMutex);
	for (auto& entry : g_deviceToHostPinnedMemoryPool) {
		cudaFreeHost(entry.second);
	}
	g_deviceToHostPinnedMemoryPool.clear();
}

// 释放特定大小的设备到主机固定内存（可选优化）
void releaseDeviceToHostPinnedMemory(size_t byteSize)
{
	std::lock_guard<std::mutex> lock(g_deviceToHostMemoryPoolMutex);
	auto it = g_deviceToHostPinnedMemoryPool.find(byteSize);
	if (it != g_deviceToHostPinnedMemoryPool.end()) {
		cudaFreeHost(it->second);
		g_deviceToHostPinnedMemoryPool.erase(it);
		std::cout << "[DEBUG] 已释放固定内存，大小: " << (byteSize / 1024) << "KB" << std::endl;
	}
}

// 获取内存池状态信息
void getDeviceToHostMemoryPoolInfo(int* poolSize, size_t* totalMemoryBytes)
{
	std::lock_guard<std::mutex> lock(g_deviceToHostMemoryPoolMutex);
	*poolSize = static_cast<int>(g_deviceToHostPinnedMemoryPool.size());
	*totalMemoryBytes = 0;
	for (const auto& entry : g_deviceToHostPinnedMemoryPool) {
		*totalMemoryBytes += entry.first;
	}
}

// 清理所有的内存池
void cleanupAllPinnedMemoryPools()
{
	cleanupHostToDevicePinnedMemoryPool();
	cleanupDeviceToHostPinnedMemoryPool();
}

ExceptionStatus copyFloatDeviceToHostByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	BEGIN_WRAP_TRTAPI
		if (nodeIndex < 0 || nodeIndex >= ptr->numTensors) {
			return ExceptionStatus::OccurredTRT;
		}

	const char* tensorName = ptr->tensorNames[nodeIndex];
	// 获取张量形状信息
	nvinfer1::Dims dims = ptr->context->getTensorShape(tensorName);
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpyAsync(data, ptr->dataBuffer[nodeIndex], size * sizeof(float), cudaMemcpyDeviceToHost, ptr->stream));
	CHECKCUDA(cudaStreamSynchronize(ptr->stream));
	END_WRAP_TRTAPI
}

ExceptionStatus nvinferDelete(NvinferStruct* ptr)
{
	BEGIN_WRAP_TRTAPI
		// 清理GPU内存缓冲区
		for (int32_t i = 0; i < ptr->numTensors; ++i)
		{
			if (ptr->dataBuffer[i]) {
				CHECKCUDA(cudaFree(ptr->dataBuffer[i]);)
					ptr->dataBuffer[i] = nullptr;
			}
		}
	delete[] ptr->dataBuffer;
	ptr->dataBuffer = nullptr;

	// 清理张量名称
	if (ptr->tensorNames) {
		for (int32_t i = 0; i < ptr->numTensors; ++i) {
			if (ptr->tensorNames[i]) {
				delete[] ptr->tensorNames[i];
				ptr->tensorNames[i] = nullptr;
			}
		}
		delete[] ptr->tensorNames;
		ptr->tensorNames = nullptr;
	}

	// 清理资源
	delete ptr->context;
	delete ptr->engine;
	delete ptr->runtime;
	CHECKCUDA(cudaStreamDestroy(ptr->stream));
	delete ptr;
	END_WRAP_TRTAPI
}

ExceptionStatus getBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int* dimLength, int* dims)
{
	BEGIN_WRAP_TRTAPI
		// 获取张量形状信息
		nvinfer1::Dims shape_d = ptr->context->getTensorShape(nodeName);
	*dimLength = shape_d.nbDims;
	for (int i = 0; i < *dimLength; ++i)
	{
		*dims++ = shape_d.d[i];
	}
	END_WRAP_TRTAPI
}

ExceptionStatus getBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int* dimLength, int* dims)
{
	BEGIN_WRAP_TRTAPI
		if (nodeIndex < 0 || nodeIndex >= ptr->numTensors) {
			return ExceptionStatus::OccurredTRT;
		}

	const char* tensorName = ptr->tensorNames[nodeIndex];
	// 获取张量形状信息
	nvinfer1::Dims shape_d = ptr->context->getTensorShape(tensorName);
	*dimLength = shape_d.nbDims;
	for (int i = 0; i < *dimLength; ++i)
	{
		*dims++ = shape_d.d[i];
	}
	END_WRAP_TRTAPI
}

// 释放映射的主机内存
ExceptionStatus freeMappedHostMemory(float* mappedPtr)
{
	BEGIN_WRAP_TRTAPI
		if (mappedPtr) {
			CHECKCUDA(cudaFreeHost(mappedPtr));
		}
	END_WRAP_TRTAPI
}


