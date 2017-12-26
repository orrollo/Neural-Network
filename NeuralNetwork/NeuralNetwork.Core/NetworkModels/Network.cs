using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Core.ActivationFunctions;

namespace NeuralNetwork.Core.NetworkModels
{
	public abstract class Network
	{
		#region -- Properties --

	    public List<Neuron> InputLayer { get; set; }
		public List<List<Neuron>> HiddenLayers { get; set; }
		public List<Neuron> OutputLayer { get; set; }
		#endregion

	    protected Type HiddenActivationType;
        protected Type OutputActivationType;

		#region -- Globals --
		private static readonly Random Random = new Random();
		#endregion

		#region -- Constructor --

	    protected Network(Type hiddenActivationType = null, Type outputActivationType = null)
	    {
	        HiddenActivationType = hiddenActivationType ?? typeof (Sigmoid);
	        OutputActivationType = outputActivationType ?? typeof (Sigmoid);
			CreateLayers();
		}

	    private void CreateLayers()
	    {
	        InputLayer = new List<Neuron>();
	        HiddenLayers = new List<List<Neuron>>();
	        OutputLayer = new List<Neuron>();
	    }

        protected Network(int inputSize, int[] hiddenSizes, int outputSize, Type hiddenActivationType = null, Type outputActivationType = null)
            : this(hiddenActivationType, outputActivationType)
	    {
	        CreateNeurons(inputSize, hiddenSizes, outputSize);
	    }

        protected void CreateNeurons(int inputSize, int[] hiddenSizes, int outputSize) 
	    {
	        for (var i = 0; i < inputSize; i++)
	            InputLayer.Add(new Neuron());

	        var firstHiddenLayer = new List<Neuron>();
	        for (var i = 0; i < hiddenSizes[0]; i++)
	            firstHiddenLayer.Add(new Neuron(InputLayer, HiddenActivationType));

	        HiddenLayers.Add(firstHiddenLayer);

	        for (var i = 1; i < hiddenSizes.Length; i++)
	        {
	            var hiddenLayer = new List<Neuron>();
	            for (var j = 0; j < hiddenSizes[i]; j++)
	                hiddenLayer.Add(new Neuron(HiddenLayers[i - 1], HiddenActivationType));
	            HiddenLayers.Add(hiddenLayer);
	        }

	        for (var i = 0; i < outputSize; i++)
	            OutputLayer.Add(new Neuron(HiddenLayers.Last(), OutputActivationType));
	    }

	    #endregion

		#region -- Training --

	    protected void ForwardPropagate(params double[] inputs)
		{
			var i = 0;
			InputLayer.ForEach(a => a.Value = inputs[i++]);
			HiddenLayers.ForEach(a => a.ForEach(b => b.CalculateValue()));
			OutputLayer.ForEach(a => a.CalculateValue());
		}

	    public double[] Compute(params double[] inputs)
		{
			ForwardPropagate(inputs);
			return OutputLayer.Select(a => a.Value).ToArray();
		}

        protected double CalculateError(params double[] targets)
        {
            var i = 0;
            return OutputLayer.Sum(a => Math.Abs(a.CalculateError(targets[i++])));
        }

        public abstract void Train(List<DataSet> dataSets, TrainParams.TrainParams trainParams);
        
        #endregion

		#region -- Helpers --
		public static double GetRandom()
		{
			return 2 * Random.NextDouble() - 1;
		}
		#endregion

	}

    #region -- Enum --

    #endregion
}