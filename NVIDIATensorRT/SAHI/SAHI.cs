using NVIDIATensorRT.Deploy;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace NVIDIATensorRT.SAHI
{
    /// <summary>  
    /// 超图切片推理  
    /// </summary>  
    public class SAHI : TensorRT
    {
        /// <summary>
        /// 超图切片
        /// </summary>
        /// <param name="widthSize">固定宽度尺寸</param>
        /// <param name="heightSize">固定高度尺寸</param>
        /// <param name="inputMat">原图片</param>
        /// <param name="overlap">重叠率</param>
        /// <returns>切片集合</returns>
        public static SliceUpMatList SupergraphPartitionFixed(int widthSize, int heightSize, Mat inputMat, float overlap = 0.1f)
        {
            Stopwatch swTotal = Stopwatch.StartNew();
            Stopwatch sw = new Stopwatch();

            var resultList = new SliceUpMatList();

            // --- Step1: 极小图直接填充 ---
            sw.Restart();
            int strideX = Math.Max(1, (int)(widthSize * (1 - overlap)));
            int strideY = Math.Max(1, (int)(heightSize * (1 - overlap)));
            if (inputMat.Width <= widthSize && inputMat.Height <= heightSize)
            {
                var single = new Mat(new Size(widthSize, heightSize), inputMat.Type(), Scalar.Black);
                using (var dstRoi = new Mat(single, new Rect(0, 0, inputMat.Width, inputMat.Height)))
                {
                    inputMat.CopyTo(dstRoi);
                }
                resultList.Add(new SliceMat
                {
                    Peak = new Point(0, 0),
                    Image = single
                });

                sw.Stop();
                Console.WriteLine($"SupergraphPartitionFixed (极小图填充): {sw.ElapsedMilliseconds} ms (总: {sw.ElapsedMilliseconds} ms)");
                return resultList;
            }
            sw.Stop();
            var timeStrideAndCheck = sw.ElapsedMilliseconds;

            // --- Step2: 计算 xStarts / yStarts ---
            sw.Restart();
            var xStarts = new List<int>(Math.Max(1, (inputMat.Width + strideX - 1) / Math.Max(1, strideX)));
            var yStarts = new List<int>(Math.Max(1, (inputMat.Height + strideY - 1) / Math.Max(1, strideY)));

            for (int x = 0; x < inputMat.Width; x += strideX)
                xStarts.Add(x);
            int lastX = Math.Max(0, inputMat.Width - widthSize);
            if (xStarts.Count == 0 || xStarts[xStarts.Count - 1] != lastX)
                xStarts.Add(lastX);

            for (int y = 0; y < inputMat.Height; y += strideY)
                yStarts.Add(y);
            int lastY = Math.Max(0, inputMat.Height - heightSize);
            if (yStarts.Count == 0 || yStarts[yStarts.Count - 1] != lastY)
                yStarts.Add(lastY);
            sw.Stop();
            var timeCalcStarts = sw.ElapsedMilliseconds;

            // --- Step3: 生成切片 (最重) ---
            sw.Restart();
            int xCount = xStarts.Count;
            int yCount = yStarts.Count;
            int total = xCount * yCount;
            var results = new SliceMat[total];

            Parallel.For(0, total, i =>
            {
                int iy = i / xCount;
                int ix = i % xCount;
                int x0 = xStarts[ix];
                int y0 = yStarts[iy];
                int w = Math.Min(widthSize, inputMat.Width - x0);
                int h = Math.Min(heightSize, inputMat.Height - y0);
                var output = new Mat(new Size(widthSize, heightSize), inputMat.Type(), Scalar.Black);
                var srcRect = new Rect(x0, y0, w, h);
                using (var srcRoi = new Mat(inputMat, srcRect))
                {
                    if (w == widthSize && h == heightSize)
                    {
                        srcRoi.CopyTo(output);
                    }
                    else
                    {
                        var dstRect = new Rect(0, 0, w, h);
                        using (var dstRoi = new Mat(output, dstRect))
                        {
                            srcRoi.CopyTo(dstRoi);
                        }
                    }
                }

                results[i] = new SliceMat
                {
                    Peak = new Point(x0, y0),
                    Image = output
                };
            });
            sw.Stop();
            var timeParallelCrop = sw.ElapsedMilliseconds;

            // --- Step4: 结果组装 ---
            sw.Restart();
            resultList.AddRange(results);
            sw.Stop();
            var timeResultAssemble = sw.ElapsedMilliseconds;

            swTotal.Stop();
            Console.WriteLine(
      $"SupergraphPartitionFixed 耗时分布: 尺寸检查+stride计算={timeStrideAndCheck} ms, " +
      $"计算 starts={timeCalcStarts} ms, " +
      $"Parallel 切片={timeParallelCrop} ms, " +
      $"结果组装={timeResultAssemble} ms, " +
      $"总耗时={swTotal.ElapsedMilliseconds} ms");

            return resultList;
        }
    }
}
