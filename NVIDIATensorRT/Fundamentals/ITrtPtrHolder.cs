using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NVIDIATensorRT
{
    /// <summary>
    /// 本机指针
    /// </summary>
    public interface ITrtPtrHolder
    {
        /// <summary>
        /// TensorRT数据指针
        /// </summary>
        IntPtr TrtPtr { get; }
    }
}
