using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NVIDIATensorRT.Deploy;
using NVIDIATensorRT.Custom;
using OpenCvSharp;
using System.Diagnostics;

namespace Test
{
    internal unsafe class Program
    {
        static void Main(string[] args)
        {
            TensorRT tensorRT = new TensorRT(@"D:\train18\weights\best.engine", new NVIDIATensorRT.Dims(new int[] { 5, 3, 640, 640 }));
            tensorRT.NVIDIALogEvent += ((s) =>
            {
                Console.WriteLine(s);
            });
            InputTensor input = new InputTensor();
            OutputTensor output = new OutputTensor();
            tensorRT.Tensor("images", "output0", ref input, ref output);
            string[] files = Directory.GetFiles(@"D:\\AI\\Ai\\demo712\\cut1", "*.jpg");
            Mat img = Cv2.ImRead(files[0]);
            Mat image1 = Cv2.ImRead(files[3]);
            Mat image2 = Cv2.ImRead(files[2]);
            Stopwatch stopwatch = new Stopwatch();
            for (int i = 0; i < 500 - 5; i += 5)
            {
                List<Mat> images = new List<Mat>();
                for (int j = 0; j < 5; j++)
                {
                    images.Add(Cv2.ImRead(files[i + j]));
                }
                var t = tensorRT.InFerencePtr(ref input, ref output, images);
                stopwatch.Start();
                var tr = tensorRT.BatchParseYoloOutput((float*)t, input, output, new Size(640, 640), @"C:\Users\Forever\Downloads\Compressed\data.yaml");
                stopwatch.Stop();
                Console.WriteLine(stopwatch.ElapsedMilliseconds + "");
                stopwatch.Reset();
            }
            Console.ReadKey();
        }
    }
}
