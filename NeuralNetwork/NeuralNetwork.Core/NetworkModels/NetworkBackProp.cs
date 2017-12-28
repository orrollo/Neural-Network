using System;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetwork.Core.Params;

namespace NeuralNetwork.Core.NetworkModels
{
    //public class NetworkBackProp : Network
    //{
    //    public NetworkBackProp(Type hiddenActivationType = null, Type outputActivationType = null)
    //        : base(hiddenActivationType, outputActivationType)
    //    {
    //    }

    //    public NetworkBackProp(int inputSize, int[] hiddenSizes, int outputSize, Type hiddenActivationType = null, Type outputActivationType = null) 
    //        : base(inputSize, hiddenSizes, outputSize, hiddenActivationType, outputActivationType)
    //    {
    //    }

    //    public override void Train(List<DataSet> dataSets, TrainParams trainParams)
    //    {
    //        var bpParams = trainParams as BackPropTrainParams;
    //        if (bpParams == null) throw new ArgumentException("params must be <BackPropTrainParams> object");
    //        if (bpParams.Training == TrainingType.Epoch) 
    //            TrainByEpochs(dataSets, bpParams);
    //        else if (bpParams.Training == TrainingType.MinimumError)
    //            TrainByError(dataSets, bpParams);
    //        else
    //            throw new ArgumentException("unknown training type");
    //    }

    //    public void TrainByEpochs(List<DataSet> dataSets, BackPropTrainParams bpp)
    //    {
    //        for (var i = 0; i < bpp.NumEpochs; i++)
    //        {
    //            foreach (var dataSet in dataSets)
    //            {
    //                ForwardPropagate(dataSet.Values);
    //                BackPropagate(bpp, dataSet.Targets);
    //            }
    //        }
    //        bpp.ResultEpochs = bpp.NumEpochs;
    //    }

    //    public void TrainByError(List<DataSet> dataSets, BackPropTrainParams bpp)
    //    {
    //        var error = 1.0;
    //        var numEpochs = 0;
    //        var rnd = new Random();
    //        var src = new List<DataSet>(dataSets);

    //        var count = src.Count;
    //        while (error > bpp.MinimumError && numEpochs < int.MaxValue)
    //        {
    //            if ((numEpochs % count) == 0)
    //            {
    //                for (int i = 0; i < src.Count; i++)
    //                {
    //                    int j = rnd.Next(src.Count);
    //                    if (i == j) continue;
    //                    var set = src[i];
    //                    src[i] = src[j];
    //                    src[j] = set;
    //                }
    //            }
    //            foreach (var dataSet in src)
    //            {
    //                ForwardPropagate(dataSet.Values);
    //                BackPropagate(bpp, dataSet.Targets);
    //            }
    //            error = CalcErrorForData(src);
    //            numEpochs++;
    //        }
    //        bpp.ResultEpochs = numEpochs;
    //    }

    //    private double CalcErrorForData(List<DataSet> src)
    //    {
    //        var error = 0.0;
    //        foreach (var dataSet in src)
    //        {
    //            ForwardPropagate(dataSet.Values);
    //            error += CalculateError(dataSet.Targets);
    //        }
    //        error /= src.Count;
    //        return error;
    //    }

    //    private void BackPropagate(BackPropTrainParams bpp, params double[] targets)
    //    {
    //        var i = 0;
    //        double learnRate = bpp.LearnRate, momentum = bpp.Momentum;

    //        OutputLayer.ForEach(a => a.CalculateGradient(targets[i++]));
            
    //        HiddenLayers.Reverse();
    //        if (bpp.UseMultiThreading)
    //        {
    //            HiddenLayers.ForEach(a => Parallel.ForEach(a, bpp.ParallelOptionsOptions, b => b.CalculateGradient()));
    //            HiddenLayers.ForEach(a => Parallel.ForEach(a, bpp.ParallelOptionsOptions, b => b.UpdateWeights(learnRate, momentum)));
    //        }
    //        else
    //        {
    //            HiddenLayers.ForEach(a => a.ForEach(b => b.CalculateGradient()));
    //            HiddenLayers.ForEach(a => a.ForEach(b => b.UpdateWeights(learnRate, momentum)));
    //        }
    //        HiddenLayers.Reverse();

    //        if (bpp.UseMultiThreading)
    //        {
    //            Parallel.ForEach(OutputLayer, bpp.ParallelOptionsOptions, a => a.UpdateWeights(learnRate, momentum));
    //        }
    //        else
    //        {
    //            OutputLayer.ForEach(a => a.UpdateWeights(learnRate, momentum));
    //        }
    //    }
    //}
}