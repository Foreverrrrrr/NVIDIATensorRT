using System;

namespace NVIDIATensorRT
{
    /// <summary>
    /// DisposableObject ICvPtrHolder组合抽象类
    /// </summary>
    public abstract class DisposableTrtObject : DisposableObject, ITrtPtrHolder
    {
        /// <summary>
        /// 数据指针
        /// </summary>
        protected IntPtr ptr;

        /// <summary>
        /// 默认构造函数
        /// </summary>
        protected DisposableTrtObject()
            : this(true)
        {
        }

        /// <summary>
        /// 构造函数，带指针参数
        /// </summary>
        /// <param name="ptr">指针</param>
        protected DisposableTrtObject(IntPtr ptr)
            : this(ptr, true)
        {
        }

        /// <summary>
        /// 构造函数，带是否启用释放标志
        /// </summary>
        /// <param name="isEnabledDispose">是否允许释放资源</param>
        protected DisposableTrtObject(bool isEnabledDispose)
            : this(IntPtr.Zero, isEnabledDispose)
        {
        }

        /// <summary>
        /// 构造函数，带指针和是否启用释放标志
        /// </summary>
        /// <param name="ptr">指针</param>
        /// <param name="isEnabledDispose">是否允许释放资源</param>
        protected DisposableTrtObject(IntPtr ptr, bool isEnabledDispose)
            : base(isEnabledDispose)
        {
            this.ptr = ptr;
        }

        /// <summary>
        /// 释放非托管资源
        /// </summary>
        protected override void DisposeUnmanaged()
        {
            ptr = IntPtr.Zero;
            base.DisposeUnmanaged();
        }

        /// <summary>
        /// OpenCV 结构的本机指针
        /// </summary>
        public IntPtr TrtPtr
        {
            get
            {
                ThrowIfDisposed();
                return ptr;
            }
        }
    }

}
