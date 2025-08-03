using System;
using System.Runtime.InteropServices;

namespace NVIDIATensorRT.Internal
{
    public static partial class NativeMethods
    {
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus onnxToEngine(ref sbyte onnxFile, int memorySize);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus onnxToEngineDynamicShape(ref sbyte onnxFile, int memorySize, ref sbyte nodeName, ref int minShapes, ref int optShapes, ref int maxShapes);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus nvinferInit(ref sbyte engineFile, out IntPtr returnNvinferPtr);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus nvinferInitDynamicShape(ref sbyte engineFile, int maxBatahSize, out IntPtr ptr);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus copyFloatHostToDeviceByName(IntPtr nvinferPtr, ref sbyte nodeName, ref float dataArray);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus copyFloatHostToDeviceByIndex(IntPtr nvinferPtr, int nodeIndex, ref float dataArray);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus setBindingDimensionsByName(IntPtr ptr, ref sbyte nodeName, int nbDims, ref int dims);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus setBindingDimensionsByIndex(IntPtr ptr, int nodeIndex, int nbDims, ref int dims);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus tensorRtInfer(IntPtr nvinferPtr);
        [DllImport(tensorrt_dll_path, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe ExceptionStatus copyFloatDeviceToHostByName(IntPtr nvinferPtr, sbyte* nodeName, float* returnData, UIntPtr elementCount);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus copyFloatDeviceToHostByIndex(IntPtr nvinferPtr, int nodeIndex, ref float rereturnDataArrayturnData);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus nvinferDelete(IntPtr nvinferPtr);
        [DllImport(tensorrt_dll_path, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe ExceptionStatus getBindingDimensionsByName(IntPtr nvinferPtr, sbyte* nodeName, out int dimLength, int* dims);
        [DllImport(tensorrt_dll_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public extern static ExceptionStatus getBindingDimensionsByIndex(IntPtr nvinferPtr, int index, out int dimLength, ref int dims);
        [DllImport(tensorrt_dll_path, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe ExceptionStatus copyFloatHostToDeviceByName(IntPtr nvinferPtr, sbyte* nodeName, float* data, int elementCount);
        [DllImport(tensorrt_dll_path, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe ExceptionStatus copyFloatHostToDeviceByNameZeroCopy(IntPtr nvinferPtr, sbyte* nodeName, float* data, UIntPtr elementCount);
        [DllImport(tensorrt_dll_path, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe ExceptionStatus copyFloatDeviceToHostZeroCopy(IntPtr nvinferPtr, sbyte* nodeName, ref IntPtr hostData, UIntPtr elementCount);
        [DllImport(tensorrt_dll_path, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cleanupAllPinnedMemoryPools();

        [DllImport(tensorrt_dll_path, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cleanupDeviceToHostPinnedMemoryPool();
        [DllImport(tensorrt_dll_path, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cleanupHostToDevicePinnedMemoryPool();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void CopyCompleteCallback(IntPtr hostData, IntPtr userData,double elapsedMs);

        [DllImport(tensorrt_dll_path, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern ExceptionStatus copyFloatDeviceToHostAsync(
            IntPtr nvinferPtr,
            string nodeName,
            UIntPtr elementCount,
            CopyCompleteCallback callback,
            IntPtr userData
        );
    }
}
