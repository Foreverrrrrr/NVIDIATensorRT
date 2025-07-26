using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NVIDIATensorRT.Internal;
using OpenCvSharp;

namespace NVIDIATensorRT.Custom
{
    /// <summary>
    /// TensorRT 推理引擎封装类
    /// </summary>
    public class Nvinfer : DisposableTrtObject
    {
        private readonly object _syncLock = new object();

        /// <summary>
        /// 将 ONNX 模型文件转换为 TensorRT 引擎文件
        /// </summary>
        /// <param name="modelPath">ONNX 模型文件的路径</param>
        /// <param name="memorySize">分配的内存大小</param>
        public static void OnnxToEngine(string modelPath, int memorySize)
        {
            sbyte[] modelPathSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(modelPath));
            HandleException.Handler(NativeMethods.onnxToEngine(ref modelPathSbyte[0], memorySize));
        }

        /// <summary>
        /// 将 ONNX 模型文件转换为 TensorRT 引擎文件，支持动态形状
        /// </summary>
        /// <param name="modelPath">ONNX 模型文件的路径</param>
        /// <param name="memorySize">分配的内存大小</param>
        /// <param name="nodeName">节点名称</param>
        /// <param name="minShapes">最小形状</param>
        /// <param name="optShapes">最优形状</param>
        /// <param name="maxShapes">最大形状</param>
        public static void OnnxToEngine(string modelPath, int memorySize, string nodeName, Dims minShapes, Dims optShapes, Dims maxShapes)
        {
            sbyte[] modelPathSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(modelPath));
            sbyte[] nodeNameSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(nodeName));
            int[] min = minShapes.ToArray();
            int[] opt = optShapes.ToArray();
            int[] max = maxShapes.ToArray();
            HandleException.Handler(NativeMethods.onnxToEngineDynamicShape(ref modelPathSbyte[0], memorySize, ref nodeNameSbyte[0], ref min[0], ref opt[0], ref max[0]));
        }

        /// <summary>
        /// 默认构造函数
        /// </summary>
        public Nvinfer() { }

        /// <summary>
        /// 使用给定的路径初始化 TensorRT 引擎
        /// </summary>
        /// <param name="modelPath">TensorRT 引擎文件路径</param>
        public Nvinfer(string modelPath)
        {
            sbyte[] modelPathSbyte = (sbyte[])((Array)System.Text.Encoding.UTF8.GetBytes(modelPath));
            HandleException.Handler(NativeMethods.nvinferInit(ref modelPathSbyte[0], out ptr));
        }

        /// <summary>
        /// 使用给定的模型路径和最大批次大小初始化 TensorRT 引擎
        /// </summary>
        /// <param name="modelPath">TensorRT 引擎文件路径</param>
        /// <param name="maxBatahSize">最大批次大小</param>
        public Nvinfer(string modelPath, int maxBatahSize)
        {
            sbyte[] modelPathSbyte = (sbyte[])((Array)System.Text.Encoding.UTF8.GetBytes(modelPath));
            HandleException.Handler(NativeMethods.nvinferInitDynamicShape(ref modelPathSbyte[0], maxBatahSize, out ptr));
        }

        /// <summary>
        /// 获取绑定的输入/输出张量维度
        /// </summary>
        /// <param name="index">绑定的节点索引</param>
        /// <returns>返回一个包含维度信息的 Dims 对象</returns>
        public Dims GetBindingDimensions(int index)
        {
            int l = 0;
            int[] d = new int[8];
            HandleException.Handler(NativeMethods.getBindingDimensionsByIndex(ptr, index, out l, ref d[0]));
            return new Dims(l, d);
        }

        /// <summary>
        /// 根据节点名称获取绑定的输入/输出张量的维度信息
        /// </summary>
        /// <param name="nodeName">节点名称</param>
        /// <returns>返回一个包含维度信息的 Dims 对象</returns>
        public unsafe Dims GetBindingDimensions(string nodeName)
        {
            sbyte[] nodeNameSbyte = Encoding.UTF8.GetBytes(nodeName + '\0').Select(b => (sbyte)b).ToArray();
            int dimLength = 0;
            int[] dimsArray = new int[8];
            fixed (sbyte* namePtr = nodeNameSbyte)
            fixed (int* dimsPtr = dimsArray)
            {
                HandleException.Handler(NativeMethods.getBindingDimensionsByName(ptr, namePtr, out dimLength, dimsPtr));
            }
            return new Dims(dimLength, dimsArray);
        }

        /// <summary>
        /// 设置绑定的输入/输出张量的维度信息
        /// </summary>
        /// <param name="index">绑定的节点索引</param>
        /// <param name="dims">新的维度信息</param>
        public void SetBindingDimensions(int index, Dims dims)
        {
            int l = dims.nbDims;
            int[] d = dims.ToArray();
            HandleException.Handler(NativeMethods.setBindingDimensionsByIndex(ptr, index, l, ref d[0]));
        }

        /// <summary>
        /// 根据节点名称设置绑定的输入/输出张量的维度信息
        /// </summary>
        /// <param name="nodeName">节点名称</param>
        /// <param name="dims">新的维度信息</param>
        public void SetBindingDimensions(string nodeName, Dims dims)
        {
            sbyte[] nodeNameSbyte = (sbyte[])((Array)System.Text.Encoding.UTF8.GetBytes(nodeName));
            int l = dims.nbDims;
            int[] d = dims.ToArray();
            HandleException.Handler(NativeMethods.setBindingDimensionsByName(ptr, ref nodeNameSbyte[0], l, ref d[0]));
        }

        /// <summary>
        /// 加载推理输入数据
        /// </summary>
        /// <param name="nodeName">节点名称</param>
        /// <param name="data">输入数据</param>
        public void LoadInferenceData(string nodeName, float[] data)
        {
            sbyte[] nodeNameSbyte = (sbyte[])((Array)System.Text.Encoding.Default.GetBytes(nodeName));
            HandleException.Handler(NativeMethods.copyFloatHostToDeviceByName(ptr, ref nodeNameSbyte[0], ref data[0]));
        }

        /// <summary>
        /// 根据节点索引加载推理输入数据
        /// </summary>
        /// <param name="nodeIndex">节点索引</param>
        /// <param name="data">输入数据</param>
        public void LoadInferenceData(int nodeIndex, float[] data)
        {
            HandleException.Handler(NativeMethods.copyFloatHostToDeviceByIndex(ptr, nodeIndex, ref data[0]));
        }

        /// <summary>
        /// 加载推理输入数据（使用原生指针）
        /// </summary>
        /// <param name="nodeName">节点名称</param>
        /// <param name="dataPtr">数据指针</param>
        /// <param name="elementCount">数据元素数量</param>
        public void LoadInferenceData(string nodeName, IntPtr dataPtr, int elementCount)
        {
            if (dataPtr == IntPtr.Zero)
                throw new ArgumentException("输入数据指针不能为空");
            if (string.IsNullOrEmpty(nodeName))
                throw new ArgumentNullException(nameof(nodeName));
            byte[] nameBytes = Encoding.UTF8.GetBytes(nodeName + "\0");
            unsafe
            {
                fixed (byte* pName = nameBytes)
                {
                    var status = NativeMethods.copyFloatHostToDeviceByName(
                        ptr,
                        (sbyte*)pName,
                        (float*)dataPtr.ToPointer(),
                        elementCount);
                    HandleException.Handler(status);
                }
            }
        }

        /// <summary>
        /// 执行推理任务
        /// </summary>
        public void Infer()
        {
            HandleException.Handler(NativeMethods.tensorRtInfer(ptr));
        }

        /// <summary>
        /// 获取推理结果（返回浮点数组）
        /// </summary>
        /// <param name="nodeName">节点名称</param>
        /// <returns>推理结果数组</returns>
        public unsafe float[] GetInferenceResultArray(string nodeName)
        {
            sbyte[] nodeNameSbyte = Encoding.UTF8.GetBytes(nodeName + '\0').Select(b => (sbyte)b).ToArray();
            int dimLength = 0;
            Dims dims = new Dims();
            int[] dimsArray = new int[8];
            fixed (sbyte* namePtr = nodeNameSbyte)
            fixed (int* dimsPtr = dimsArray)
            {
                HandleException.Handler(NativeMethods.getBindingDimensionsByName(ptr, namePtr, out dimLength, dimsPtr));
            }
            dims = new Dims(dimLength, dimsArray);
            float[] result = new float[dims.Prod()];
            fixed (sbyte* namePtr = nodeNameSbyte)
            fixed (float* resultPtr = result)
            {
                HandleException.Handler(NativeMethods.copyFloatDeviceToHostByName(ptr, namePtr, resultPtr));
            }
            return result;
        }

        /// <summary>
        /// 获取推理结果的原生指针
        /// </summary>
        /// <param name="nodeName">节点名称</param>
        /// <param name="length">结果数据的长度</param>
        /// <returns>推理结果的指针</returns>
        public unsafe InferenceTensor GetInferenceResultPtr(string nodeName, out int length)
        {
            sbyte[] nodeNameSbyte = Encoding.UTF8.GetBytes(nodeName + '\0').Select(b => (sbyte)b).ToArray();
            int dimLength = 0;
            int[] dimsArray = new int[8];
            fixed (sbyte* namePtr = nodeNameSbyte)
            fixed (int* dimsPtr = dimsArray)
            {
                HandleException.Handler(NativeMethods.getBindingDimensionsByName(ptr, namePtr, out dimLength, dimsPtr));
            }
            Dims dims = new Dims(dimLength, dimsArray);
            length = dims.Prod();
            var result = new InferenceTensor(length);
            fixed (sbyte* namePtr = nodeNameSbyte)
            {
                HandleException.Handler(NativeMethods.copyFloatDeviceToHostByName(ptr, namePtr, result.ResultPtr));
            }
            return result;
        }

        /// <summary>
        /// 释放资源
        /// </summary>
        public void Release()
        {
            Dispose();
        }

        /// <inheritdoc />
        /// <summary>
        /// 释放托管资源
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            if (ptr != IntPtr.Zero && IsEnabledDispose)
                HandleException.Handler(NativeMethods.nvinferDelete(ptr));
            base.DisposeUnmanaged();
        }
    }
}
