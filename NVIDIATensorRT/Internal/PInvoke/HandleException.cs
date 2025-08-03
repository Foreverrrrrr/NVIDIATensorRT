using NVIDIATensorRT.Internal.PInvoke;
using System.Runtime.InteropServices;

namespace NVIDIATensorRT.Internal
{
    /// <summary>
    /// TensorRT C API 返回值异常检测处理类
    /// </summary>
    static class HandleException
    {
        /// <summary>
        /// 检查返回值是否有异常，如有异常，根据异常类型抛出对应的异常
        /// </summary>
        /// <param name="status">返回状态</param>
        public static void Handler(ExceptionStatus status)
        {
            if (ExceptionStatus.NotOccurred == status)
            {
                return;
            }
            else if (ExceptionStatus.Occurred == status)
            {
                general_exception();
            }
            else if (ExceptionStatus.OccurredTRT == status)
            {
                tensorrt_exception();
            }
            else if (ExceptionStatus.OccurredCuda == status)
            {
                cuda_exception();
            }
        }

        /// <summary>
        /// 抛出通用异常 TRTException
        /// </summary>
        /// <exception cref="OVException">通用异常</exception>
        private static void general_exception()
        {
            throw new TRTException(ExceptionStatus.Occurred, Marshal.PtrToStringAnsi(NativeMethods.trt_get_last_err_msg()));
        }

        /// <summary>
        /// 抛出 TensorRT 特定异常 TRTException
        /// </summary>
        /// <exception cref="OVException">TensorRT 异常</exception>
        private static void tensorrt_exception()
        {
            throw new TRTException(ExceptionStatus.OccurredTRT, Marshal.PtrToStringAnsi(NativeMethods.trt_get_last_err_msg()));
        }

        /// <summary>
        /// 抛出 CUDA 特定异常 TRTException
        /// </summary>
        /// <exception cref="OVException">CUDA 异常</exception>
        private static void cuda_exception()
        {
            throw new TRTException(ExceptionStatus.OccurredCuda, Marshal.PtrToStringAnsi(NativeMethods.trt_get_last_err_msg()));
        }
    }

}
