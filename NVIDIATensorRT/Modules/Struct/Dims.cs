using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NVIDIATensorRT
{
    /// <summary>
    /// 维度结构体
    /// </summary>
    public unsafe struct Dims
    {
        /// <summary>
        /// 维度的最大数量限制
        /// </summary>
        public const int MAX_DIMS = 8;

        /// <summary>
        /// 当前有效维度数目
        /// </summary>
        public int nbDims;

        /// <summary>
        /// 维度数组，固定长度为MAX_DIMS（8）
        /// </summary>
        public fixed int d[MAX_DIMS];

        /// <summary>
        /// 构造函数，指定维度数量和维度数据数组
        /// </summary>
        /// <param name="leng">维度数量</param>
        /// <param name="data">维度数据数组，长度需不小于leng</param>
        /// <exception cref="ArgumentException">当data长度小于leng时抛出异常</exception>
        public Dims(int leng, int[] data)
        {
            nbDims = leng;
            if (data.Length < leng)
                throw new ArgumentException("Data length less than nbDims");
            for (int i = 0; i < MAX_DIMS; i++)
                d[i] = (i < leng) ? data[i] : 0;
        }

        /// <summary>
        /// 构造函数，使用可变参数数组初始化维度，维度数量即为数组长度
        /// </summary>
        /// <param name="data">维度数据数组</param>
        public Dims(params int[] data) : this(data.Length, data) { }

        /// <summary>
        /// 计算所有维度的乘积（即元素总数）
        /// </summary>
        /// <returns>返回所有维度值的乘积</returns>
        public int Prod()
        {
            int p = 1;
            for (int i = 0; i < nbDims; i++)
                p *= d[i];
            return p;
        }

        /// <summary>
        /// 返回维度数组的字符串表示形式，例如 "[1, 3, 224, 224]"
        /// </summary>
        /// <returns>维度的字符串</returns>
        public override string ToString()
        {
            var dims = new int[nbDims];
            for (int i = 0; i < nbDims; i++)
                dims[i] = d[i];
            return $"[{string.Join(", ", dims)}]";
        }

        /// <summary>
        /// 将固定长度的维度数据转换为普通数组并返回
        /// </summary>
        /// <returns>返回当前有效维度的整型数组</returns>
        public unsafe int[] ToArray()
        {
            int[] result = new int[nbDims];
            for (int i = 0; i < nbDims; i++)
            {
                result[i] = d[i];
            }
            return result;
        }
    }
}
