using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NVIDIATensorRT.Custom
{
    /// <summary>
    /// 推理输出非托管内存
    /// </summary>
    public unsafe class InferenceTensor : IDisposable
    {
        /// <summary>
        /// 非托管内存指针
        /// </summary>
        public IntPtr UnmanagedPtr { get; private set; }

        /// <summary>
        /// float 类型指针，指向推理输出数据
        /// </summary>
        public float* ResultPtr { get; private set; }

        /// <summary>
        /// 推理结果元素个数
        /// </summary>
        public int Length { get; private set; }

        /// <summary>
        /// 创建推理张量对象，并分配指定数量的 float 类型非托管内存
        /// </summary>
        /// <param name="length">需要分配的元素数量</param>
        public InferenceTensor(int length)
        {
            Length = length;
            UnmanagedPtr = Marshal.AllocHGlobal(length * sizeof(float));
            ResultPtr = (float*)UnmanagedPtr;
        }

        /// <summary>
        /// 释放非托管内存资源，防止内存泄漏
        /// </summary>
        public void Dispose()
        {
            if (UnmanagedPtr != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(UnmanagedPtr);
                UnmanagedPtr = IntPtr.Zero;
                ResultPtr = null;
            }
        }
    }
}
