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

// �������豸���㿽���ڴ��
std::unordered_map<size_t, void*> g_hostToDevicePinnedMemoryPool;
std::mutex g_hostToDeviceMemoryPoolMutex;

// �豸���������㿽���ڴ��
std::unordered_map<size_t, void*> g_deviceToHostPinnedMemoryPool;
std::mutex g_deviceToHostMemoryPoolMutex;

// @brief ������onnxģ��תΪtensorrt�е�engine��ʽ�������浽����
// @param onnx_file_path_wchar onnxģ�ͱ��ص�ַ
// @param engine_file_path_wchar engineģ�ͱ��ص�ַ
// @param type ���ģ�;��ȣ�
ExceptionStatus onnxToEngine(const char* onnxFile, int memorySize) {
	BEGIN_WRAP_TRTAPI
	// ��·����Ϊ�������ݸ�����
	std::string path(onnxFile);
	std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
	std::string modelPath = path.substr(0, iPos);//��ȡ�ļ�·��
	std::string modelName = path.substr(iPos, path.length() - iPos);//��ȡ����׺���ļ���
	std::string modelName_ = modelName.substr(0, modelName.rfind("."));//��ȡ������׺���ļ�����
	std::string engineFile = modelPath + modelName_ + ".engine";
	//std::cout << model_name << std::endl;
	//std::cout << model_name_ << std::endl;
	//std::cout << model_path << std::endl;
	//std::cout << engine_file << std::endl;

	// ����������ȡcuda�ں�Ŀ¼�Ի�ȡ����ʵ��
	// ���ڴ���config��network��engine����������ĺ�����
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
	// ������������
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// ����onnx�����ļ�
	// tensorRTģ����
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	// onnx�ļ�������
	// ��onnx�ļ������������rensorRT����ṹ
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
	// ����onnx�ļ�
	parser->parseFromFile(onnxFile, 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// ������������
	// �������������ö���
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// ����������ռ��С��
	config->setMaxWorkspaceSize(1024 * 1024 * memorySize);
	// ����ģ���������
	config->setFlag(nvinfer1::BuilderFlag::kFP16);
	// ������������
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// ��������ǹ���浽����
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream filePtr(engineFile, std::ios::binary);
	if (!filePtr) {
		std::cerr << "could not open plan output file" << std::endl;
		return ExceptionStatus::Occurred;
	}
	// ��ģ��ת��Ϊ�ļ�������
	nvinfer1::IHostMemory* modelStream = engine->serialize();
	// ���ļ����浽����
	filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	// ���ٴ����Ķ���
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
		// ��·����Ϊ�������ݸ�����
		std::string path(onnxFile);
	std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
	std::string modelPath = path.substr(0, iPos);//��ȡ�ļ�·��
	std::string modelName = path.substr(iPos, path.length() - iPos);//��ȡ����׺���ļ���
	std::string modelName_ = modelName.substr(0, modelName.rfind("."));//��ȡ������׺���ļ�����
	std::string engineFile = modelPath + modelName_ + ".engine";
	//std::cout << model_name << std::endl;
	//std::cout << model_name_ << std::endl;
	//std::cout << model_path << std::endl;
	//std::cout << engine_file << std::endl;

	// ����������ȡcuda�ں�Ŀ¼�Ի�ȡ����ʵ��
	// ���ڴ���config��network��engine����������ĺ�����
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
	// ������������
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	// ����onnx�����ļ�
	// tensorRTģ����
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	// ������������
	// �������������ö���
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// ����������ռ��С��
	config->setMaxWorkspaceSize(1024 * 1024 * memorySize);
	// ����ģ���������
	config->setFlag(nvinfer1::BuilderFlag::kFP16);

	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(minShapes[0], minShapes[1], minShapes[2], minShapes[3]));
	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(optShapes[0], optShapes[1], optShapes[2], optShapes[3]));
	profile->setDimensions(nodeName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxShapes[0], maxShapes[1], maxShapes[2], maxShapes[3]));

	config->addOptimizationProfile(profile);

	// onnx�ļ�������
	// ��onnx�ļ������������rensorRT����ṹ
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
	// ����onnx�ļ�
	parser->parseFromFile(onnxFile, 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");

	// ������������
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	// ��������ǹ���浽����
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream filePtr(engineFile, std::ios::binary);
	if (!filePtr) {
		std::cerr << "could not open plan output file" << std::endl;
		return ExceptionStatus::Occurred;
	}
	// ��ģ��ת��Ϊ�ļ�������
	nvinfer1::IHostMemory* modelStream = engine->serialize();
	// ���ļ����浽����
	filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	// ���ٴ����Ķ���
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
	// �Զ����Ʒ�ʽ��ȡ�ʼ�
	std::ifstream filePtr(engineFile, std::ios::binary);
	if (!filePtr.good()) {
		std::cerr << "�ļ��޷��򿪣���ȷ���ļ��Ƿ���ã�" << std::endl;
		dup_last_err_msg("Model file reading error, please confirm if the file exists or if the format is correct.");
		return ExceptionStatus::Occurred;
	}

	size_t size = 0;
	filePtr.seekg(0, filePtr.end);	// ����ָ����ļ�ĩβ��ʼ�ƶ�0���ֽ�
	size = filePtr.tellg();	// ���ض�ָ���λ�ã���ʱ��ָ���λ�þ����ļ����ֽ���
	filePtr.seekg(0, filePtr.beg);	// ����ָ����ļ���ͷ��ʼ�ƶ�0���ֽ�
	char* modelStream = new char[size];
	filePtr.read(modelStream, size);
	// �ر��ļ�
	filePtr.close();

	// ����������Ľṹ�壬��ʼ������
	NvinferStruct* p = new NvinferStruct();
	// ��ʼ�������л�����
	CHECKTRT(p->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
	CHECKTRT(p->runtime->setErrorRecorder(&gRecorder));
	// ��ʼ����������
	CHECKTRT(p->engine = p->runtime->deserializeCudaEngine(modelStream, size));
	// ����������
	CHECKTRT(p->context = p->engine->createExecutionContext());
	CHECKCUDA(cudaStreamCreate(&(p->stream)));
	CHECKTRT(int numNode = p->engine->getNbBindings());
	// ����gpu���ݻ�����
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
		case nvinfer1::DataType::kFLOAT: size *= 4; break;  // ��ȷΪ���� float
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

// @brief ��ȡ����engineģ�ͣ�����ʼ��NvinferStruct�����仺��ռ�
ExceptionStatus nvinferInitDynamicShape(const char* engineFile, int maxBatahSize, NvinferStruct** ptr) {
	BEGIN_WRAP_TRTAPI
		initLibNvInferPlugins(nullptr, "");
		// �Զ����Ʒ�ʽ��ȡ�ʼ�
		std::ifstream filePtr(engineFile, std::ios::binary);
	if (!filePtr.good()) {
		std::cerr << "�ļ��޷��򿪣���ȷ���ļ��Ƿ���ã�" << std::endl;
		dup_last_err_msg("Model file reading error, please confirm if the file exists or if the format is correct.");
		return ExceptionStatus::Occurred;
	}

	size_t size = 0;
	filePtr.seekg(0, filePtr.end);	// ����ָ����ļ�ĩβ��ʼ�ƶ�0���ֽ�
	size = filePtr.tellg();	// ���ض�ָ���λ�ã���ʱ��ָ���λ�þ����ļ����ֽ���
	filePtr.seekg(0, filePtr.beg);	// ����ָ����ļ���ͷ��ʼ�ƶ�0���ֽ�
	char* modelStream = new char[size];
	filePtr.read(modelStream, size);
	// �ر��ļ�
	filePtr.close();

	// ����������Ľṹ�壬��ʼ������
	NvinferStruct* p = new NvinferStruct();
	// ��ʼ�������л�����
	CHECKTRT(p->runtime = nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
	CHECKTRT(p->runtime->setErrorRecorder(&gRecorder));
	// ��ʼ����������
	CHECKTRT(p->engine = p->runtime->deserializeCudaEngine(modelStream, size));
	// ����������
	CHECKTRT(p->context = p->engine->createExecutionContext());
	CHECKCUDA(cudaStreamCreate(&(p->stream)));
	CHECKTRT(int numNode = p->engine->getNbBindings());
	// ����gpu���ݻ�����
	p->dataBuffer = new void* [numNode];
	delete[] modelStream;

	for (int i = 0; i < numNode; i++) {
		CHECKTRT(nvinfer1::Dims dims = p->engine->getBindingDimensions(i));
		CHECKTRT(nvinfer1::DataType type = p->engine->getBindingDataType(i));
		size_t size = std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
		switch (type)
		{
		case nvinfer1::DataType::kINT32:
		case nvinfer1::DataType::kFLOAT: size *= 4; break;  // ��ȷΪ���� float
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
	// ��ȡ����ڵ�δ����Ϣ
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
		CHECKCUDA(cudaStreamSynchronize(ptr->stream)); // ȷ�������Ѵ��䵽�豸
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
	// ��ȡ����ڵ�δ����Ϣ
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
/// ����ָ��GPUȡ����CPU
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

// �������C++�Ż��汾 - ʹ����ȷ��ExceptionStatus����ֵ
ExceptionStatus WaitForInferenceCompletion(NvinferStruct* ptr, int timeoutMs = 1000)
{
	if (!ptr || !ptr->stream) {
		return ExceptionStatus::NotOccurred;
	}

	auto start = std::chrono::high_resolution_clock::now();

	while (true) {
		cudaError_t streamStatus = cudaStreamQuery(ptr->stream);
		if (streamStatus == cudaSuccess) {
			std::cout << "[DEBUG] �������ȷ��" << std::endl;
			return ExceptionStatus::NotOccurred;
		}
		else if (streamStatus != cudaErrorNotReady) {
			std::cerr << "[ERROR] ����ѯ����: " << cudaGetErrorString(streamStatus) << std::endl;
			return ExceptionStatus::OccurredCuda;
		}

		// ��鳬ʱ
		auto current = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration<double, std::milli>(current - start).count();
		if (elapsed > timeoutMs) {
			std::cout << "[WARNING] �ȴ�������ɳ�ʱ: " << elapsed << "ms" << std::endl;
			return ExceptionStatus::NotOccurred; // ��ʱ������ִ��
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}

// ��ȡ�����쳣״̬����C#�˵��ã�
ExceptionStatus GetLastExceptionStatus(NvinferStruct* ptr)
{
	// ���CUDA����
	cudaError_t cudaErr = cudaGetLastError();
	if (cudaErr != cudaSuccess) {
		std::cout << "[DEBUG] ��⵽CUDA����: " << cudaGetErrorString(cudaErr) << std::endl;
		return ExceptionStatus::OccurredCuda;
	}

	// ���TensorRT״̬����������API��
	// ����������TensorRT�ض��Ĵ�����

	return ExceptionStatus::NotOccurred;
}

// ��������Ż����䷽��
ExceptionStatus copyFloatDeviceToHostZeroCopy(
	NvinferStruct* ptr,
	const char* nodeName,
	float** hostData,
	size_t elementCount)
{
	try {
		auto start = std::chrono::high_resolution_clock::now();

		// 1. ���ȵȴ��������
		std::cout << "[DEBUG] ��ʼ�ȴ��������..." << std::endl;
		auto wait_start = std::chrono::high_resolution_clock::now();

		ExceptionStatus waitResult = WaitForInferenceCompletion(ptr, 2000);
		if (waitResult != ExceptionStatus::NotOccurred) {
			std::cout << "[ERROR] �ȴ��������ʧ�ܣ�״̬��: " << static_cast<int>(waitResult) << std::endl;
			return waitResult;
		}

		auto after_wait = std::chrono::high_resolution_clock::now();
		auto wait_time = std::chrono::duration<double, std::milli>(after_wait - wait_start).count();
		std::cout << "[DEBUG] ������ɵȴ���ʱ: " << wait_time << "ms" << std::endl;

		// 2. ��ȡ�ڵ�����
		int nodeIndex = ptr->engine->getBindingIndex(nodeName);
		if (nodeIndex < 0) {
			std::cerr << "[ERROR] ��Ч�ڵ���: " << nodeName << std::endl;
			return ExceptionStatus::OccurredTRT;
		}

		size_t byteSize = elementCount * sizeof(float);

		// 3. ����̶��ڴ�
		void* pinnedMem = nullptr;
		cudaError_t err = cudaHostAlloc(&pinnedMem, byteSize, cudaHostAllocMapped);
		if (err != cudaSuccess) {
			std::cerr << "[ERROR] cudaHostAllocʧ��: " << cudaGetErrorString(err) << std::endl;
			return ExceptionStatus::OccurredCuda;
		}

		// 4. ִ�п������ݴ���
		auto copy_start = std::chrono::high_resolution_clock::now();

		err = cudaMemcpyAsync(pinnedMem, ptr->dataBuffer[nodeIndex], byteSize,
			cudaMemcpyDeviceToHost, ptr->stream);
		if (err != cudaSuccess) {
			std::cerr << "[ERROR] cudaMemcpyAsyncʧ��: " << cudaGetErrorString(err) << std::endl;
			cudaFreeHost(pinnedMem);
			return ExceptionStatus::OccurredCuda;
		}

		err = cudaStreamSynchronize(ptr->stream);
		if (err != cudaSuccess) {
			std::cerr << "[ERROR] cudaStreamSynchronizeʧ��: " << cudaGetErrorString(err) << std::endl;
			cudaFreeHost(pinnedMem);
			return ExceptionStatus::OccurredCuda;
		}

		auto copy_end = std::chrono::high_resolution_clock::now();
		auto copy_time = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
		auto total_time = std::chrono::duration<double, std::milli>(copy_end - start).count();

		*hostData = static_cast<float*>(pinnedMem);

		std::cout << "[DEBUG] �Ż��������: "
			<< "����ȴ�=" << wait_time << "ms, "
			<< "������=" << copy_time << "ms, "
			<< "�ܺ�ʱ=" << total_time << "ms" << std::endl;

		// ���㴿�����ٶ�
		double pureTransferSpeedMBps = (byteSize / 1024.0 / 1024.0) / (copy_time / 1000.0);
		std::cout << "[DEBUG] �������ٶ�: " << pureTransferSpeedMBps << "MB/s" << std::endl;

		// ��������
		if (wait_time > total_time * 0.8) {
			std::cout << "[ANALYSIS] ��Ҫ��ʱ������ȴ�(" << wait_time << "ms)������ʹ���첽ģʽ" << std::endl;
		}
		if (copy_time > 20) {
			std::cout << "[ANALYSIS] ������ʱ��ϳ�(" << copy_time << "ms)�����ܴ���PCIeƿ��" << std::endl;
		}

		return ExceptionStatus::NotOccurred;
	}
	catch (const std::exception& e) {
		std::cerr << "[EXCEPTION] C++�쳣: " << e.what() << std::endl;
		return ExceptionStatus::Occurred;
	}
	catch (...) {
		std::cerr << "[EXCEPTION] δ֪C++�쳣" << std::endl;
		return ExceptionStatus::Occurred;
	}
}

// ��������Ƿ���ɵļ򵥰汾����C#���ã�
bool IsInferenceComplete(NvinferStruct* ptr)
{
	if (!ptr || !ptr->stream) {
		return true;
	}

	cudaError_t streamStatus = cudaStreamQuery(ptr->stream);
	bool isComplete = (streamStatus == cudaSuccess);

	std::cout << "[DEBUG] ����״̬��ѯ: " << (isComplete ? "���" : "������") << std::endl;
	return isComplete;
}


// �ص�����ǩ��
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
		std::cout << "[DEBUG-ASYNC] �ڵ�����: " << nodeIndex
			<< ", ���ݴ�С: " << (byteSize / 1024) << "KB" << std::endl;
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
				std::cout << "[DEBUG-ASYNC] �����µ�˫����ԣ���С: " << (byteSize / 1024) << "KB" << std::endl;
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
				std::cout << "[DEBUG-ASYNC] ˫���������ɣ���ʱ: " << allocTime << "ms, "
					<< "�����ܻ���: " << g_doubleBufferPool.size() << "��, "
					<< "���ڴ�: " << (totalBufferAllocated / 1024 / 1024) << "MB" << std::endl;
			}
			else {
				std::cout << "[DEBUG-ASYNC] ��������˫�����" << std::endl;
			}
			buffers = &it->second;
		}

		auto poolEnd = std::chrono::high_resolution_clock::now();
		auto poolTime = std::chrono::duration<double, std::milli>(poolEnd - poolStart).count();
		cudaError_t streamStatus = cudaStreamQuery(ptr->stream);
		std::cout << "[DEBUG-ASYNC] GPU��״̬: " <<
			(streamStatus == cudaSuccess ? "����" :
				streamStatus == cudaErrorNotReady ? "æµ" :
				cudaGetErrorString(streamStatus)) << std::endl;

		void* devicePtr = ptr->dataBuffer[nodeIndex];
		int writeIndex = buffers->index;

		std::cout << "[DEBUG-ASYNC] ʹ�û���������: " << writeIndex
			<< ", �豸ָ��: " << devicePtr
			<< ", ��������: " << buffers->hostBuf[writeIndex] << std::endl;
		auto copyStart = std::chrono::high_resolution_clock::now();
		std::cout << "[DEBUG-ASYNC] ��ʼ�첽�ڴ濽��..." << std::endl;

		CHECKCUDA(cudaMemcpyAsync(
			buffers->hostBuf[writeIndex],
			devicePtr,
			byteSize,
			cudaMemcpyDeviceToHost,
			ptr->stream));

		auto copyLaunch = std::chrono::high_resolution_clock::now();
		auto copyLaunchTime = std::chrono::duration<double, std::milli>(copyLaunch - copyStart).count();
		std::cout << "[DEBUG-ASYNC] �첽���������ɣ���ʱ: " << copyLaunchTime << "ms" << std::endl;

		// 5. �����첽�ص�
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

		std::cout << "[DEBUG-ASYNC] �����첽�ص�..." << std::endl;

		CHECKCUDA(cudaLaunchHostFunc(ptr->stream, [](void* userData) {
			auto* d = reinterpret_cast<CallbackData*>(userData);

			auto endTime = std::chrono::high_resolution_clock::now();
			auto copyElapsed = std::chrono::duration<double, std::milli>(endTime - d->startTime).count();
			auto totalElapsed = std::chrono::duration<double, std::milli>(endTime - d->funcStartTime).count();

			// ���㴫���ٶ�
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
		std::cerr << "[EXCEPTION-ASYNC] C++�쳣: " << e.what() << std::endl;
		return ExceptionStatus::Occurred;
	}
	END_WRAP_TRTAPI
}


// �����������豸���ڴ��
void cleanupHostToDevicePinnedMemoryPool()
{
	std::lock_guard<std::mutex> lock(g_hostToDeviceMemoryPoolMutex);
	for (auto& entry : g_hostToDevicePinnedMemoryPool) {
		cudaFreeHost(entry.second);
	}
	g_hostToDevicePinnedMemoryPool.clear();
}

// �����豸���������ڴ��
void cleanupDeviceToHostPinnedMemoryPool()
{
	std::lock_guard<std::mutex> lock(g_deviceToHostMemoryPoolMutex);
	for (auto& entry : g_deviceToHostPinnedMemoryPool) {
		cudaFreeHost(entry.second);
	}
	g_deviceToHostPinnedMemoryPool.clear();
}

// �������е��ڴ��
void cleanupAllPinnedMemoryPools()
{
	cleanupHostToDevicePinnedMemoryPool();
	cleanupDeviceToHostPinnedMemoryPool();
}

ExceptionStatus copyFloatDeviceToHostByIndex(NvinferStruct* ptr, int nodeIndex, float* data)
{
	BEGIN_WRAP_TRTAPI
	// ��ȡ����ڵ�δ����Ϣ
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
	// ��ȡ����ڵ�δ����Ϣ
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
	// ��ȡ����ڵ�δ����Ϣ
	CHECKTRT(nvinfer1::Dims shape_d = ptr->context->getBindingDimensions(nodeIndex));
	*dimLength = shape_d.nbDims;
	for (int i = 0; i < *dimLength; ++i)
	{
		*dims++ = shape_d.d[i];
	}
	END_WRAP_TRTAPI
}


