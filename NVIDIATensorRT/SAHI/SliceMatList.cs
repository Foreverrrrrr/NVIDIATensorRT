using OpenCvSharp;
using System;
using System.Collections.Generic;

namespace NVIDIATensorRT.SAHI
{
    /// <summary>
    /// 切片集合
    /// </summary>
    public class SliceUpMatList : List<SliceMat>, IDisposable
    {
        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// 释放资源
        /// </summary>
        /// <param name="disposing"></param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var slice in this)
                {
                    slice.Dispose();
                }
                this.Clear();
            }
        }


        ~SliceUpMatList()
        {
            Dispose(false);
        }
    }

    /// <summary>
    /// 切片结构
    /// </summary>
    public class SliceMat : IDisposable
    {
        /// <summary>
        /// 切片顶点坐标
        /// </summary>
        public Point Peak { get; set; }

        /// <summary>
        /// 切图
        /// </summary>
        public Mat Image { get; set; }

        ~SliceMat()
        {
            Dispose(false);
        }

        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                Image?.Dispose();
            }
        }
    }
}
