using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.NetworkModels;

namespace NeuralNetwork.Tests
{
    using NUnit.Framework;

    [TestFixture]
    public class NetworkXorTest
    {
        [Test]
        public void XorTest()
        {
            var nnet = new Network(2, new int[] {2}, 1);
            var ds = BuildXorDataSets();

            nnet.Train(ds, 5000);

            Assert.AreEqual(Math.Abs(nnet.Compute(0, 0)[0] - 0.0) < 0.1, true);
            Assert.AreEqual(Math.Abs(nnet.Compute(0, 1)[0] - 1.0) < 0.1, true);
            Assert.AreEqual(Math.Abs(nnet.Compute(1, 0)[0] - 1.0) < 0.1, true);
            Assert.AreEqual(Math.Abs(nnet.Compute(1, 1)[0] - 0.0) < 0.1, true);
        }

        private static List<DataSet> BuildXorDataSets()
        {
            var ds = new List<DataSet>();
            ds.Add(new DataSet(new double[] {0, 0}, new double[] {0}));
            ds.Add(new DataSet(new double[] {0, 1}, new double[] {1}));
            ds.Add(new DataSet(new double[] {1, 0}, new double[] {1}));
            ds.Add(new DataSet(new double[] {1, 1}, new double[] {0}));
            return ds;
        }
    }
}
