using Hybridizer.Runtime.CUDAImports;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

namespace Cudatimus {
	class Program {
		private static readonly IEnumerable<(long, long)> TestCases = new List<(long, long)>()
		{
			(1, 1_000), (1, 2_000), (1, 3_000), (1, 4_000), (1, 5_000), (1, 6_000), (1, 7_000), (1, 8_000), (1, 9_000),
			(1, 10_000), (1, 20_000), (1, 30_000), (1, 40_000), (1, 50_000), (1, 60_000), (1, 70_000), (1, 80_000), (1, 90_000),
			(1, 100_000), (1, 200_000), (1, 300_000), (1, 400_000), (1, 500_000), (1, 600_000), (1, 700_000), (1, 800_000), (1, 900_000),
			(1, 1_000_000), (1, 10_000_000), (1_000_000, 2_000_000), (10_000_000, 20_000_000)
		};
		//static void Main(string[] args)
		//{
		//	cudaDeviceProp prop;
		//	cuda.GetDeviceProperties(out prop, 0);
		//	HybRunner runner = HybRunner.Cuda().SetDistrib(prop.multiProcessorCount * 16, 128);

		//	// create a wrapper object to call GPU methods instead of C#
		//	dynamic wrapped = runner.Wrap(new PrimeGeneratorGPU());

		//	foreach (var testCase in TestCases)
		//	{
		//		var time = MeasureTime(() => {
		//			wrapped.GeneratePrimes(testCase.Item1, testCase.Item2);
		//		});
		//	}
		//}

		[EntryPoint("run")]
		public static void Run(int N, double[] a, double[] b)
		{
			Parallel.For(0, N, i => { a[i] += b[i]; });
		}

		static void Main(string[] args)
		{
			// 268 MB allocated on device -- should fit in every CUDA compatible GPU
			int N = 1024 * 1024 * 16;
			double[] acuda = new double[N];
			double[] adotnet = new double[N];

			double[] b = new double[N];

			Random rand = new Random();

			//Initialize acuda et adotnet and b by some doubles randoms, acuda and adotnet have same numbers. 
			for (int i = 0; i < N; ++i)
			{
				acuda[i] = rand.NextDouble();
				adotnet[i] = acuda[i];
				b[i] = rand.NextDouble();
			}

			cudaDeviceProp prop;
			cuda.GetDeviceProperties(out prop, 0);
			HybRunner runner = HybRunner.Cuda().SetDistrib(prop.multiProcessorCount * 16, 128);

			// create a wrapper object to call GPU methods instead of C#
			dynamic wrapped = runner.Wrap(new Program());

			// run the method on GPU
			wrapped.Run(N, acuda, b);

			// run .Net method
			Run(N, adotnet, b);

			// verify the results
			for (int k = 0; k < N; ++k)
			{
				if (acuda[k] != adotnet[k])
					Console.Out.WriteLine("ERROR !");
			}
			Console.Out.WriteLine("DONE");
		}

		private static long MeasureTime(Action action)
		{
			var sw = new Stopwatch();
			sw.Start();
			try
			{
				action.Invoke();
			}
			catch (Exception e)
			{
				Console.Write("Exception Occured: {0}", e.Message);
			}
			sw.Stop();

			Console.WriteLine("Took {0} secs.", (sw.ElapsedMilliseconds / 1000f));
			return sw.ElapsedTicks;
		}
	}
}
