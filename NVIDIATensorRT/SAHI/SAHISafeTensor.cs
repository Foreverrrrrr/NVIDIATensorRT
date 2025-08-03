using System;
using System.Runtime.InteropServices;

namespace NVIDIATensorRT.SAHI
{

    /// <summary>
    /// 代表一个安全的张量（SAHITensor）对象，封装了内存中的数据并提供对该数据的高效访问
    /// 通过使用 `GCHandle` 锁定数据，内存GC回收，调用 `Dispose` 方法释放资源
    /// </summary>
    public unsafe class SAHISafeTensor : IDisposable
    {
        private readonly float[] _buffer;

        private readonly GCHandle _handle;

        /// <summary>
        /// 张量的维度
        /// </summary>
        public int[] Dimensions { get; }

        /// <summary>
        /// 批次大小
        /// </summary>
        public int BatchSize => Dimensions[0];

        /// <summary>
        /// 通道数
        /// </summary>
        public int Channels => Dimensions[1];

        /// <summary>
        /// 图像的高度
        /// </summary>
        public int Height => Dimensions[2];

        /// <summary>
        /// 图像的宽度
        /// </summary>
        public int Width => Dimensions[3];

        /// <summary>
        /// 切片集合
        /// </summary>
        public SliceUpMatList Mats { get; private set; }

        /// <summary>
        /// 张量数据内存锁定
        /// </summary>
        /// <param name="data">张量数据数组</param>
        /// <param name="dims">张量的维度数组</param>
        /// <param name="slices">切片集合</param>
        public SAHISafeTensor(float[] data, int[] dims, SliceUpMatList slices)
        {
            Mats = slices;
            _buffer = data;
            Dimensions = dims;
            _handle = GCHandle.Alloc(_buffer, GCHandleType.Pinned);//内存锁定
        }

        /// <summary>
        /// 获取已锁定内存中的数据指针
        /// </summary>
        public IntPtr DataPtr => _handle.AddrOfPinnedObject();

        /// <summary>
        /// 获取张量的总元素数
        /// </summary>
        public int Length => _buffer.Length;

        /// <summary>
        /// 获取张量数据Span 对象
        /// </summary>
        public Span<float> AsSpan() => _buffer.AsSpan();

        /// <summary>
        /// 释放 占用的资源，解除对内存的锁定
        /// </summary>
        public void Dispose()
        {
            Mats?.Dispose();
            _handle.Free();
            GC.SuppressFinalize(this);
        }
    }
}
