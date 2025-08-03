using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace NVIDIATensorRT
{
    /// <summary>
    /// 管理内存的基类
    /// </summary>
    public abstract class DisposableObject : IDisposable
    {
        /// <summary>
        /// 获取或设置使用 cvSetData 分配的句柄
        /// </summary>
        protected GCHandle DataHandle { get; private set; }

        private volatile int disposeSignaled = 0;

        /// <summary>
        /// 获取一个值，指示该实例是否已被释放
        /// </summary>
        public bool IsDisposed { get; protected set; }

        /// <summary>
        /// 获取或设置一个值，指示是否允许释放该实例的资源
        /// </summary>
        public bool IsEnabledDispose { get; set; }

        /// <summary>
        /// 获取或设置通过 AllocMemory 分配的内存地址
        /// </summary>
        protected IntPtr AllocatedMemory { get; set; }

        /// <summary>
        /// 获取或设置已分配内存的字节大小
        /// </summary>
        protected long AllocatedMemorySize { get; set; }

        /// <summary>
        /// 默认构造函数
        /// </summary>
        protected DisposableObject()
            : this(true)
        {
        }

        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="isEnabledDispose">是否允许通过垃圾回收器释放该类的资源，true 表示允许</param>
        protected DisposableObject(bool isEnabledDispose)
        {
            IsDisposed = false;
            IsEnabledDispose = isEnabledDispose;
            AllocatedMemory = IntPtr.Zero;
            AllocatedMemorySize = 0;
        }

        /// <summary>
        /// 释放资源的公共方法。
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// 释放资源的核心方法
        /// </summary>
        /// <param name="disposing">
        /// 如果为 true，表示由用户代码直接或间接调用，可释放托管和非托管资源；
        /// 如果为 false，表示由运行时终结器调用，只能释放非托管资源
        /// </param>
        protected virtual void Dispose(bool disposing)
        {
            if (Interlocked.Exchange(ref disposeSignaled, 1) != 0)
            {
                return;
            }
            IsDisposed = true;
            if (IsEnabledDispose)
            {
                if (disposing)
                {
                    DisposeManaged();
                }
                DisposeUnmanaged();
            }
        }

        /// <summary>
        /// 垃圾回收
        /// </summary>
        ~DisposableObject()
        {
            Dispose(false);
        }

        /// <summary>
        /// 释放托管资源
        /// </summary>
        protected virtual void DisposeManaged()
        {
        }

        /// <summary>
        /// 释放非托管资源
        /// </summary>
        protected virtual void DisposeUnmanaged()
        {
            if (DataHandle.IsAllocated)
            {
                DataHandle.Free();
            }
            if (AllocatedMemorySize > 0)
            {
                GC.RemoveMemoryPressure(AllocatedMemorySize);
                AllocatedMemorySize = 0;
            }
            if (AllocatedMemory != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(AllocatedMemory);
                AllocatedMemory = IntPtr.Zero;
            }
        }

        /// <summary>
        /// 固定内存对象
        /// </summary>
        /// <param name="obj">要固定的对象。</param>
        /// <returns>返回固定后的句柄</returns>
        protected internal GCHandle AllocGCHandle(object obj)
        {
            if (obj is null)
                throw new ArgumentNullException(nameof(obj));
            if (DataHandle.IsAllocated)
                DataHandle.Free();
            DataHandle = GCHandle.Alloc(obj, GCHandleType.Pinned);
            return DataHandle;
        }

        /// <summary>
        /// 分配指定大小的内存
        /// </summary>
        /// <param name="size">要分配的字节大小</param>
        /// <returns>返回分配的内存指针</returns>
        protected IntPtr AllocMemory(int size)
        {
            if (size <= 0)
                throw new ArgumentOutOfRangeException(nameof(size));
            if (AllocatedMemory != IntPtr.Zero)
                Marshal.FreeHGlobal(AllocatedMemory);
            AllocatedMemory = Marshal.AllocHGlobal(size);
            NotifyMemoryPressure(size);
            return AllocatedMemory;
        }

        /// <summary>
        /// 通知垃圾回收器当前分配了多少非托管内存
        /// </summary>
        /// <param name="size">分配的内存大小（byte）</param>
        protected void NotifyMemoryPressure(long size)
        {
            if (!IsEnabledDispose)
                return;
            if (size == 0)
                return;
            if (size <= 0)
                throw new ArgumentOutOfRangeException(nameof(size));
            if (AllocatedMemorySize > 0)
                GC.RemoveMemoryPressure(AllocatedMemorySize);
            AllocatedMemorySize = size;
            GC.AddMemoryPressure(size);
        }

        public void ThrowIfDisposed()
        {
            if (IsDisposed)
                throw new ObjectDisposedException(GetType().FullName);
        }
    }

}
