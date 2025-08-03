using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static NVIDIATensorRT.Custom.Nvinfer;

namespace NVIDIATensorRT.Deploy
{
    /// <summary>
    /// 封装上下文，保证 OutputTensor / Mat / 回调 不会被GC
    /// </summary>
   public class CallbackContext
    {
        public OutputTensor Tensor;
        public Mat Image;
        public CopyReasoningBack ManagedCallback;
        public GCHandle Handle; 
    }
}
