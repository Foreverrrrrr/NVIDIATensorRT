using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NVIDIATensorRT.Deploy
{
    /// <summary>
    /// 目标框结构
    /// </summary>
    public struct RectF
    {
        public RectF(float x, float y, float width, float height)
        {
            X = x;
            Y = y;
            Width = width;
            Height = height;
        }

        /// <summary>左上角X坐标</summary>
        public float X { get; set; }

        /// <summary>左上角Y坐标</summary>
        public float Y { get; set; }

        /// <summary>矩形宽度</summary>
        public float Width { get; set; }

        /// <summary>矩形高度</summary>
        public float Height { get; set; }

        /// <summary>右下角X坐标（计算属性）</summary>
        public float X2 => X + Width;

        /// <summary>右下角Y坐标（计算属性）</summary>
        public float Y2 => Y + Height;
    }


    /// <summary>
    /// 检测结果数据类
    /// </summary>
    public class DetectionResult
    {
        /// <summary>类别ID</summary>
        public int ClassId { get; set; }

        /// <summary>类别描述</summary>
        public string ClassDescribe { get; set; }

        /// <summary>检测置信度（0-1）</summary>
        public float Confidence { get; set; }

        /// <summary>原始图像坐标系下的边界框</summary>
        public RectF BoundingBox { get; set; }
    }
}
