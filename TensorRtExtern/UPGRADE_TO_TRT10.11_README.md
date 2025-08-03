# TensorRT 8.6 升级到 10.11 版本升级说明

## 主要 API 变更

### 1. 废弃的 API (TensorRT 8.6)
以下API在TensorRT 10.11中已被废弃：
- `getBindingIndex()` - 通过绑定名称获取索引
- `getBindingDimensions()` - 通过绑定索引获取维度
- `setBindingDimensions()` - 通过绑定索引设置维度
- `getNbBindings()` - 获取绑定数量
- `enqueueV2()` - 执行推理（版本2）

### 2. 新的 API (TensorRT 10.11)
使用以下新API替代：
- `getNbIOTensors()` - 获取输入输出张量数量
- `getIOTensorName()` - 获取指定索引的张量名称
- `getTensorShape()` - 获取张量形状
- `setInputShape()` - 设置输入张量形状
- `getTensorDataType()` - 获取张量数据类型
- `getTensorIOMode()` - 获取张量I/O模式
- `setTensorAddress()` - 设置张量内存地址
- `enqueueV3()` - 执行推理（版本3）
- `allInputDimensionsSpecified()` - 检查是否所有输入维度已指定

## 结构体变更

### NvinferStruct 新增字段：
```cpp
typedef struct tensorRT_nvinfer {
    // ... 原有字段 ...
    char** tensorNames;     // 张量名称数组（新增）
    int32_t numTensors;     // 张量数量（新增）
} NvinferStruct;
```

## 主要函数变更

### 1. 初始化函数
- 使用 `getNbIOTensors()` 替代 `getNbBindings()`
- 使用张量名称缓存机制，避免重复查找
- 添加张量名称存储和管理

### 2. 数据传输函数
- 所有 `copyFloat*ByName()` 函数改用张量名称查找机制
- 使用新的 `getTensorShape()` API获取张量形状
- 保持与原API的兼容性

### 3. 形状设置函数
- `setBindingDimensionsByName()` 改用 `setInputShape()`
- `setBindingDimensionsByIndex()` 通过张量名称间接调用

### 4. 推理函数
- `tensorRtInfer()` 改用 `enqueueV3()`
- 添加输入维度检查 `allInputDimensionsSpecified()`
- 使用 `setTensorAddress()` 设置张量地址

### 5. 清理函数
- 添加张量名称内存清理
- 保持原有GPU内存清理机制

## 新增功能函数

### 辅助函数
```cpp
// 查找张量索引的辅助函数
int32_t findTensorIndex(NvinferStruct* ptr, const char* nodeName);

// 获取张量数量
int32_t getTensorCount(NvinferStruct* ptr);

// 获取指定索引的张量名称
const char* getTensorNameByIndex(NvinferStruct* ptr, int32_t index);

// 检查张量是否为输入张量
bool isTensorInput(NvinferStruct* ptr, const char* tensorName);

// 检查张量是否为输出张量
bool isTensorOutput(NvinferStruct* ptr, const char* tensorName);
```

## 兼容性说明

1. **API兼容性**: 保持了原有的外部函数接口，C#代码无需修改
2. **功能兼容性**: 所有原有功能均已迁移到新API
3. **性能改进**: 使用张量名称缓存机制，避免重复查找，提升性能
4. **错误处理**: 增强了错误检查机制，提供更好的调试信息

## 注意事项

1. **编译环境**: 确保使用TensorRT 10.11 SDK进行编译
2. **CUDA版本**: 确保CUDA版本与TensorRT 10.11兼容
3. **内存管理**: 新增了张量名称的内存管理，确保正确释放
4. **调试信息**: 保留了详细的调试输出，便于问题排查

## 迁移检查清单

- [x] 更新 `NvinferStruct` 结构体
- [x] 替换所有废弃的绑定API
- [x] 实现张量名称缓存机制
- [x] 更新推理执行API
- [x] 添加新的辅助函数
- [x] 更新内存管理机制
- [x] 保持外部API兼容性
- [x] 添加错误检查和调试信息

升级完成后，建议进行充分测试以确保所有功能正常工作。
