#include "tensorrt_extern.h"
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include "ErrorRecorder.h"
#include "NvInferPlugin.h"          
#include "NvInferPluginUtils.h"  
#include "common.h"
#include <unordered_map>

// 主机到设备的零拷贝内存池
std::unordered_map<size_t, void*> g_hostToDevicePinnedMemoryPool;
std::mutex g_hostToDeviceMemoryPoolMutex;

// 设备到主机的零拷贝内存池
std::unordered_map<size_t, void*> g_deviceToHostPinnedMemoryPool;
std::mutex g_deviceToHostMemoryPoolMutex;

// @brief 将本地onnx模型转为tensorrt中的engine格式，并保存到本地
// @param onnx_file_path_wchar onnx模型本地地址
// @param engine_file_path_wchar engine模型本地地址
// @param type 输出模型精度，
ExceptionStatus onnxToEngine(const char* onnxFile, int memorySize) {
	BEGIN_WRAP_TRTAPI
	// 将路径作为参数传递给函数
	std::string path(onnxFile);
	std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
	std::string modelPath = path.substr(0, iPos);//获取文件路径
	std::string modelName = path.substr(iPos, path.length() - iPos);//获取带后缀的文件名
	std::string modelName_ = modelName.substr(0, modelName.rfind("."));//获取不带后缀的文件名名
	std::string engineFile = modelPath + modelName_ + ".engine";
	//std::cout << model_name << std::endl;
	//std::cout << model_name_ << std::endl;
	//std::cout << model_path << std::endl;
	//std::cout << engine_file << std::endl;

	// 构建器，获取cuda内核目录以获取最快的实现
	// 用于创建config、network、engine的其他对象的核心类
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
	// 定义网络属性
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// 解析onnx网络文件
	// tensorRT模型类
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	// onnx文件解析类
	// 将onnx文件解析，并填充rensorRT网络结构
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
	// 解析onnx文件
	parser->parseFromFile(onnxFile, 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// 创建推理引擎
	// 创建生成器配置对象。
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// 设置最大工作空间大小。
	config->setMaxWorkspaceSize(1024 * 1024 * memorySize);
	// 设置模型输出精度
	config->setFlag(nvinfer1::BuilderFlag::kFP16);
	// 创建推理引擎
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// 将推理银枪保存到本地
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream filePtr(engineFile, std::ios::binary);
	if (!filePtr) {
		std::cerr << "could not open plan output file" << std::endl;
		return ExceptionStatus::Occurred;
	}
	// 将模型转化为文件流数据
	nvinfer1::IHostMemory* modelStream = engine->serialize();
	// 将文件保存到本地
	filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	// 销毁创建的对象
	modelStream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
	END_WRAP_TRTAPI
}

ExceptionStatus onnxToEngineDynamicShape(const char* onnxFile, int memorySize, const char* nodeName,
	int* minShapes, int* optShapes, int* maxShapes) 
{
	BEGIN_WRAP_TRTAPI
		// 将路径作为参数传递给函数
		std::string path(onnxFile);
	std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
	std::string modelPath = path.substr(0, iPos);//获取文件路径
	std::string modelName = path.substr(iPos, path.length() - iPos);//获取带后缀的文件名
	std::string modelName_ = modelName.substr(0, modelName.rfind("."));//获取不带后缀的文件名名
	std::string engineFile = modelPath + modelName_ + ".engine";
	//std::cout << model_name << std::endl;
	//std::cout << model_name_ << std::endl;
	//std::cout << model_path << std::endl;
	//std::cout << engine_file << std::endl;

	// 构建器，获取cuda内核目录以获取最快的实现
	// 用于创建config、network、engine的其他对象的核心类
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
	// 定义网络属性
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// 解析onnx网络文件
	// tensorRT模型类
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	// 创建推理引擎
	// 创建生成器配置对象。
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// 设置最大工作空间大小。
	config->setMaxWorkspaceSize(1024 * 1024 * memorySize);
	// 设置模型输出精度
	config->setFlag(nvinfer1::BuilderFlag::kFP16);

	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(minShapes[0], minShapes[1], minShapes[2], minShapes[3]));
	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(optShapes[0], optShapes[1], optShapes[2], optShapes[3]));
	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxShapes[0], maxShapes[1], maxShapes[2], maxShapes[3]));

	config->addOptimizationProfile(profile);

	// onnx文件解析类
	// 将onnx文件解析，并填充rensorRT网络结构
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
	// 解析onnx文件
	parser->parseFromFile(onnxFile, 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// 创建推理引擎
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// 将推理银枪保存到本地
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream filePtr(engineFile, std::ios::binary);
	if (!filePtr) {
		std::cerr << "could not open plan output file" << std::endl;
		return ExceptionStatus::Occurred;
	}
	// 将模型转化为文件流数据
	nvinfer1::IHostMemory* modelStream = engine->serialize();
	// 将文件保存到本地
	filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	// 销毁创建的对象
	modelStream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
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
	// 关闭文件
	filePtr.close();

	// 创建推理核心结构体，初始化变量
	NvinferStruct* p = new NvinferStruct();
	// 初始化反序列化引擎
	CHECKTRT(p->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
	CHECKTRT(p->runtime->setErrorRecorder(&gRecorder));
	// 初始化推理引擎
	CHECKTRT(p->engine = p->runtime->deserializeCudaEngine(modelStream, size));
	// 创建上下文
	CHECKTRT(p->context = p->engine->createExecutionContext());
	CHECKCUDA(cudaStreamCreate(&(p->stream)));
	CHECKTRT(int numNode = p->engine->getNbBindings());
	// 创建gpu数据缓冲区
	p->dataBuffer = new void* [numNode];
	delete[] modelStream;

	for (int i = 0; i < numNode; i++) {
		CHECKTRT(nvinfer1::Dims dims = p->engine->getBindingDimensions(i));
		CHECKTRT(nvinfer1::DataType type = p->engine->getBindingDataType(i));
		std::vector<int> shape(dims.d, dims.d + dims.nbDims);
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
	// 关闭文件
	filePtr.close();

	// 创建推理核心结构体，初始化变量
	NvinferStruct* p = new NvinferStruct();
	// 初始化反序列化引擎
	CHECKTRT(p->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
	CHECKTRT(p->runtime->setErrorRecorder(&gRecorder));
	// 初始化推理引擎
	CHECKTRT(p->engine = p->runtime->deserializeCudaEngine(modelStream, size));
	// 创建上下文
	CHECKTRT(p->context = p->engine->createExecutionContext());
	CHECKCUDA(cudaStreamCreate(&(p->stream)));
	CHECKTRT(int numNode = p->engine->getNbBindings());
	// 创建gpu数据缓冲区
	p->dataBuffer = new void* [numNode];
	delete[] modelStream;

	for (int i = 0; i < numNode; i++) {
		CHECKTRT(nvinfer1::Dims dims = p->engine->getBindingDimensions(i));
		CHECKTRT(nvinfer1::DataType type = p->engine->getBindingDataType(i));
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
	CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
	// 获取输入节点未读信息
	CHECKTRT(nvinfer1::Dims dims = ptr->context->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpyAsync(ptr->dataBuffer[nodeIndex], data, size * sizeof(float), cudaMemcpyHostToDevice, ptr->stream));
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
		CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
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
		void* devicePtr = ptr->dataBuffer[nodeIndex];
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
	// 获取输入节点未读信息
	CHECKTRT(nvinfer1::Dims dims = ptr->context->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpyAsync(ptr->dataBuffer[nodeIndex], data, size * sizeof(float), cudaMemcpyHostToDevice, ptr->stream));
	END_WRAP_TRTAPI
}
ExceptionStatus setBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int nbDims, int* dims)
{
	CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
	switch (nbDims)
	{
		case 2:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims2(dims[0], dims[1])));
			break;
		case 3:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims3(dims[0], dims[1], dims[2])));
			break;
		case 4:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims4(dims[0], dims[1], dims[2], dims[3])));
			break;
		default:break;
	}
	END_WRAP_TRTAPI
}
ExceptionStatus setBindingDimensionsByIndex(NvinferStruct* ptr, int nodeIndex, int nbDims, int* dims)
{
	switch (nbDims)
	{
		case 2:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims2(dims[0], dims[1])));
			break;
		case 3:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims3(dims[0], dims[1], dims[2])));
			break;
		case 4:
			CHECKTRT(ptr->context->setBindingDimensions(nodeIndex, nvinfer1::Dims4(dims[0], dims[1], dims[2], dims[3])));
			break;
		default:break;
	}
	END_WRAP_TRTAPI
}

ExceptionStatus tensorRtInfer(NvinferStruct* ptr)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(ptr->context->enqueueV2((void**)ptr->dataBuffer, ptr->stream, nullptr));
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
		CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
		size_t byteSize = elementCount * sizeof(float);
		CHECKCUDA(cudaMemcpyAsync(
			data,
			ptr->dataBuffer[nodeIndex],
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
			std::cout << "[DEBUG] 推理完成确认" << std::endl;
			return ExceptionStatus::NotOccurred;
		}
		else if (streamStatus != cudaErrorNotReady) {
			std::cerr << "[ERROR] 流查询错误: " << cudaGetErrorString(streamStatus) << std::endl;
			return ExceptionStatus::OccurredCuda;
		}

		// 检查超时
		auto current = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration<double, std::milli>(current - start).count();
		if (elapsed > timeoutMs) {
			std::cout << "[WARNING] 等待推理完成超时: " << elapsed << "ms" << std::endl;
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
		std::cout << "[DEBUG] 检测到CUDA错误: " << cudaGetErrorString(cudaErr) << std::endl;
		return ExceptionStatus::OccurredCuda;
	}

	// 检查TensorRT状态（如果有相关API）
	// 这里可以添加TensorRT特定的错误检查

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
		auto start = std::chrono::high_resolution_clock::now();

		// 1. 首先等待推理完成
		std::cout << "[DEBUG] 开始等待推理完成..." << std::endl;
		auto wait_start = std::chrono::high_resolution_clock::now();

		ExceptionStatus waitResult = WaitForInferenceCompletion(ptr, 2000);
		if (waitResult != ExceptionStatus::NotOccurred) {
			std::cout << "[ERROR] 等待推理完成失败，状态码: " << static_cast<int>(waitResult) << std::endl;
			return waitResult;
		}

		auto after_wait = std::chrono::high_resolution_clock::now();
		auto wait_time = std::chrono::duration<double, std::milli>(after_wait - wait_start).count();
		std::cout << "[DEBUG] 推理完成等待耗时: " << wait_time << "ms" << std::endl;

		// 2. 获取节点索引
		int nodeIndex = ptr->engine->getBindingIndex(nodeName);
		if (nodeIndex < 0) {
			std::cerr << "[ERROR] 无效节点名: " << nodeName << std::endl;
			return ExceptionStatus::OccurredTRT;
		}

		size_t byteSize = elementCount * sizeof(float);

		// 3. 分配固定内存
		void* pinnedMem = nullptr;
		cudaError_t err = cudaHostAlloc(&pinnedMem, byteSize, cudaHostAllocMapped);
		if (err != cudaSuccess) {
			std::cerr << "[ERROR] cudaHostAlloc失败: " << cudaGetErrorString(err) << std::endl;
			return ExceptionStatus::OccurredCuda;
		}

		// 4. 执行快速数据传输
		auto copy_start = std::chrono::high_resolution_clock::now();

		err = cudaMemcpyAsync(pinnedMem, ptr->dataBuffer[nodeIndex], byteSize,
			cudaMemcpyDeviceToHost, ptr->stream);
		if (err != cudaSuccess) {
			std::cerr << "[ERROR] cudaMemcpyAsync失败: " << cudaGetErrorString(err) << std::endl;
			cudaFreeHost(pinnedMem);
			return ExceptionStatus::OccurredCuda;
		}

		err = cudaStreamSynchronize(ptr->stream);
		if (err != cudaSuccess) {
			std::cerr << "[ERROR] cudaStreamSynchronize失败: " << cudaGetErrorString(err) << std::endl;
			cudaFreeHost(pinnedMem);
			return ExceptionStatus::OccurredCuda;
		}

		auto copy_end = std::chrono::high_resolution_clock::now();
		auto copy_time = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
		auto total_time = std::chrono::duration<double, std::milli>(copy_end - start).count();

		*hostData = static_cast<float*>(pinnedMem);

		std::cout << "[DEBUG] 优化后传输分析: "
			<< "推理等待=" << wait_time << "ms, "
			<< "纯传输=" << copy_time << "ms, "
			<< "总耗时=" << total_time << "ms" << std::endl;

		// 计算纯传输速度
		double pureTransferSpeedMBps = (byteSize / 1024.0 / 1024.0) / (copy_time / 1000.0);
		std::cout << "[DEBUG] 纯传输速度: " << pureTransferSpeedMBps << "MB/s" << std::endl;

		// 性能评估
		if (wait_time > total_time * 0.8) {
			std::cout << "[ANALYSIS] 主要耗时在推理等待(" << wait_time << "ms)，建议使用异步模式" << std::endl;
		}
		if (copy_time > 20) {
			std::cout << "[ANALYSIS] 纯传输时间较长(" << copy_time << "ms)，可能存在PCIe瓶颈" << std::endl;
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


// 回调函数签名
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
		std::cout << "[DEBUG-ASYNC] 开始异步数据传输，节点: " << nodeName
			<< ", 元素数: " << elementCount << std::endl;
		CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
		if (nodeIndex < 0) {
			std::cout << "[ERROR-ASYNC] 无效节点索引: " << nodeIndex << std::endl;
			return ExceptionStatus::OccurredTRT;
		}
		size_t byteSize = elementCount * sizeof(float);
		std::cout << "[DEBUG-ASYNC] 节点索引: " << nodeIndex
			<< ", 数据大小: " << (byteSize / 1024) << "KB" << std::endl;
		struct BufferPair { void* hostBuf[2] = { nullptr, nullptr }; int index = 0; };
		static std::unordered_map<size_t, BufferPair> g_doubleBufferPool;
		static std::mutex g_mutex;
		static size_t totalBufferAllocated = 0;
		BufferPair* buffers = nullptr;
		bool newBufferCreated = false;
		auto poolStart = std::chrono::high_resolution_clock::now();

		{
			std::lock_guard<std::mutex> lock(g_mutex);
			auto it = g_doubleBufferPool.find(byteSize);
			if (it == g_doubleBufferPool.end()) {
				std::cout << "[DEBUG-ASYNC] 创建新的双缓冲对，大小: " << (byteSize / 1024) << "KB" << std::endl;
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
				std::cout << "[DEBUG-ASYNC] 双缓冲分配完成，耗时: " << allocTime << "ms, "
					<< "池中总缓冲: " << g_doubleBufferPool.size() << "个, "
					<< "总内存: " << (totalBufferAllocated / 1024 / 1024) << "MB" << std::endl;
			}
			else {
				std::cout << "[DEBUG-ASYNC] 复用现有双缓冲对" << std::endl;
			}
			buffers = &it->second;
		}

		auto poolEnd = std::chrono::high_resolution_clock::now();
		auto poolTime = std::chrono::duration<double, std::milli>(poolEnd - poolStart).count();
		cudaError_t streamStatus = cudaStreamQuery(ptr->stream);
		std::cout << "[DEBUG-ASYNC] GPU流状态: " <<
			(streamStatus == cudaSuccess ? "空闲" :
				streamStatus == cudaErrorNotReady ? "忙碌" :
				cudaGetErrorString(streamStatus)) << std::endl;

		void* devicePtr = ptr->dataBuffer[nodeIndex];
		int writeIndex = buffers->index;

		std::cout << "[DEBUG-ASYNC] 使用缓冲区索引: " << writeIndex
			<< ", 设备指针: " << devicePtr
			<< ", 主机缓冲: " << buffers->hostBuf[writeIndex] << std::endl;
		auto copyStart = std::chrono::high_resolution_clock::now();
		std::cout << "[DEBUG-ASYNC] 开始异步内存拷贝..." << std::endl;

		CHECKCUDA(cudaMemcpyAsync(
			buffers->hostBuf[writeIndex],
			devicePtr,
			byteSize,
			cudaMemcpyDeviceToHost,
			ptr->stream));

		auto copyLaunch = std::chrono::high_resolution_clock::now();
		auto copyLaunchTime = std::chrono::duration<double, std::milli>(copyLaunch - copyStart).count();
		std::cout << "[DEBUG-ASYNC] 异步拷贝启动完成，耗时: " << copyLaunchTime << "ms" << std::endl;

		// 5. 设置异步回调
		struct CallbackData {
			CopyCompleteCallback cb;
			void* buf;
			void* userData;
			std::chrono::high_resolution_clock::time_point startTime;
			std::chrono::high_resolution_clock::time_point funcStartTime;
			size_t byteSize;
			int bufferIndex;
		};

		auto* data = new CallbackData{
			callback,
			buffers->hostBuf[writeIndex],
			userData,
			copyStart,
			funcStart,
			byteSize,
			writeIndex
		};

		std::cout << "[DEBUG-ASYNC] 设置异步回调..." << std::endl;

		CHECKCUDA(cudaLaunchHostFunc(ptr->stream, [](void* userData) {
			auto* d = reinterpret_cast<CallbackData*>(userData);

			auto endTime = std::chrono::high_resolution_clock::now();
			auto copyElapsed = std::chrono::duration<double, std::milli>(endTime - d->startTime).count();
			auto totalElapsed = std::chrono::duration<double, std::milli>(endTime - d->funcStartTime).count();

			// 计算传输速度
			double transferSpeedMBps = (d->byteSize / 1024.0 / 1024.0) / (copyElapsed / 1000.0);

			std::cout << "[DEBUG-ASYNC-CALLBACK] 异步传输完成！" << std::endl;
			std::cout << "[DEBUG-ASYNC-CALLBACK] 传输分析: "
				<< "纯传输=" << copyElapsed << "ms, "
				<< "总耗时=" << totalElapsed << "ms, "
				<< "传输速度=" << transferSpeedMBps << "MB/s, "
				<< "缓冲区=" << d->bufferIndex << std::endl;
			if (d->cb) {
				std::cout << "[DEBUG-ASYNC-CALLBACK] CopyCompleteCallback回调" << std::endl;
				d->cb(d->buf, d->userData, copyElapsed);
			}

			delete d;
			}, data));

		// 6. 切换缓冲区索引
		int oldIndex = buffers->index;
		buffers->index ^= 1;
		std::cout << "[DEBUG-ASYNC] 缓冲区索引切换: " << oldIndex << " -> " << buffers->index << std::endl;

		auto funcEnd = std::chrono::high_resolution_clock::now();
		auto funcTime = std::chrono::duration<double, std::milli>(funcEnd - funcStart).count();

		std::cout << "[DEBUG-ASYNC] 异步函数执行完成: "
			<< "缓冲池=" << poolTime << "ms, "
			<< "拷贝启动=" << copyLaunchTime << "ms, "
			<< "总计=" << funcTime << "ms" << std::endl;
		std::cout << "[DEBUG-ASYNC] 异步传输已启动，等待GPU完成..." << std::endl;

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

// 清理所有的内存池
void cleanupAllPinnedMemoryPools()
{
	cleanupHostToDevicePinnedMemoryPool();
	cleanupDeviceToHostPinnedMemoryPool();
}

ExceptionStatus copyFloatDeviceToHostByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	BEGIN_WRAP_TRTAPI
	// 获取输入节点未读信息
	CHECKTRT(nvinfer1::Dims dims = ptr->context->getBindingDimensions(nodeIndex));
	std::vector<int> shape(dims.d, dims.d + dims.nbDims);
	size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	CHECKCUDA(cudaMemcpyAsync(data, ptr->dataBuffer[nodeIndex], size * sizeof(float), cudaMemcpyDeviceToHost, ptr->stream));
	CHECKCUDA(cudaStreamSynchronize(ptr->stream));
	END_WRAP_TRTAPI
}

ExceptionStatus nvinferDelete(NvinferStruct* ptr)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(int numNode = ptr->engine->getNbBindings());
	for (int i = 0; i < numNode; ++i) 
	{
		CHECKCUDA(cudaFree(ptr->dataBuffer[i]);)
		ptr->dataBuffer[i] = nullptr;
	}
	delete ptr->dataBuffer;
	ptr->dataBuffer = nullptr;
	CHECKTRT(ptr->context->destroy();)
	CHECKTRT(ptr->engine->destroy();)
	CHECKTRT(ptr->runtime->destroy();)
	CHECKCUDA(cudaStreamDestroy(ptr->stream));
	delete ptr;
	END_WRAP_TRTAPI
}

ExceptionStatus getBindingDimensionsByName(NvinferStruct* ptr, const char* nodeName, int* dimLength, int* dims)
{
	BEGIN_WRAP_TRTAPI
	CHECKTRT(int nodeIndex = ptr->engine->getBindingIndex(nodeName));
	// 获取输入节点未读信息
	CHECKTRT(nvinfer1::Dims shape_d = ptr->context->getBindingDimensions(nodeIndex));
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
	// 获取输入节点未读信息
	CHECKTRT(nvinfer1::Dims shape_d = ptr->context->getBindingDimensions(nodeIndex));
	*dimLength = shape_d.nbDims;
	for (int i = 0; i < *dimLength; ++i)
	{
		*dims++ = shape_d.d[i];
	}
	END_WRAP_TRTAPI
}


