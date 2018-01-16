using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Core.ActivationFunctions;
using NeuralNetwork.Core.Learning;
using NeuralNetwork.Core.NetworkModels;
using NUnit.Framework;

namespace NeuralNetwork.Tests
{
    [TestFixture]
    public class NetworkXorTest
    {
        [Test]
        public void BackPropSigmoidXorTest()
        {
            var nnet = new Network(2, new int[] { 2 }, 1, typeof(Sigmoid), typeof(Sigmoid));
            var learning = new BackPropLearning(nnet)
            {
                MinimumError = 0.1,
                Momentum = 0.5,
                LearnRate = 0.1
            };
            learning.TrainByError(BuildXorDataSets(), 20000);
            System.Diagnostics.Debug.WriteLine("trained in {0} epochs", learning.ResultEpochs);
            CheckResults(nnet, 0.15);
        }

        [Test]
        public void RPropSigmoidXorTest()
        {
            var nnet = new Network(2, new int[] { 2 }, 1, typeof(Sigmoid), typeof(Sigmoid));
            var learning = new RPropLearning(nnet);
            learning.Train(BuildXorDataSets());
        }

        [Test]
        public void BackPropTanhXorTest()
        {
            var nnet = new Network(2, new int[] { 2 }, 1, typeof(Tanh), typeof(Tanh));
            var learning = new BackPropLearning(nnet)
            {
                MinimumError = 0.1,
                Momentum = 0.5,
                LearnRate = 0.1
            };
            learning.TrainByError(BuildXorDataSets(), 20000);
            System.Diagnostics.Debug.WriteLine("trained in {0} epochs", learning.ResultEpochs);
            CheckResults(nnet, 0.15);
        }

        [Test]
        public void BackPropRationalSigmoidXorTest()
        {
            var nnet = new Network(2, new int[] { 2 }, 1, typeof(RationalSigmoid), typeof(RationalSigmoid));
            var learning = new BackPropLearning(nnet)
            {
                MinimumError = 0.1,
                Momentum = 0.5,
                LearnRate = 0.1
            };
            learning.TrainByError(BuildXorDataSets(), 20000);
            System.Diagnostics.Debug.WriteLine("trained in {0} epochs", learning.ResultEpochs);
            CheckResults(nnet, 0.15);
        }

        //[Test]
        //public void RPropSigmoidXorTest()
        //{
        //    var nnet = new NetworkRProp(2, new int[] { 2 }, 1, typeof(Sigmoid), typeof(Sigmoid));
        //    var trainParams = new RPropTrainParams() { NumEpochs = 500 };
        //    nnet.Train(BuildXorDataSets(), trainParams);
        //    CheckResults(nnet, 0.15);
        //}

        private static double ErrorResult(Network nnet, double result, params double[] input)
        {
            var value = nnet.Compute(input)[0];
            System.Diagnostics.Debug.WriteLine("input {0} => {1} ({2})", string.Join(";", input.Select(x => x.ToString("F2"))), value, result.ToString("F2"));
            return Math.Abs(value - result);
        }

        private static void CheckResults(Network nnet, double error)
        {
            var avg = (ErrorResult(nnet, 0, 0, 0) + ErrorResult(nnet, 1, 0, 1) + ErrorResult(nnet, 1, 1, 0) + ErrorResult(nnet, 0, 1, 1))/4;
            System.Diagnostics.Debug.WriteLine("average error " + avg.ToString("F5"));
            Assert.LessOrEqual(avg, error);
        }

        private static List<DataSet> BuildXorDataSets()
        {
            return new List<DataSet>
            {
                new DataSet(new double[] {0, 0}, new double[] {0}),
                new DataSet(new double[] {0, 1}, new double[] {1}),
                new DataSet(new double[] {1, 0}, new double[] {1}),
                new DataSet(new double[] {1, 1}, new double[] {0})
            };
        }
    }
}
