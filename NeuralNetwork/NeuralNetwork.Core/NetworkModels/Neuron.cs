using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Core.ActivationFunctions;

namespace NeuralNetwork.Core.NetworkModels
{
	public class Neuron
	{
		#region -- Properties --
		public Guid Id { get; set; }
		public List<Synapse> InputSynapses { get; set; }
		public List<Synapse> OutputSynapses { get; set; }
		public double Bias { get; set; }
		//public double BiasDelta { get; set; }
		public double Gradient { get; set; }
		public double Value { get; set; }
		#endregion

	    protected IActivationFunction ActivationFunction;

		#region -- Constructors --

	    public Neuron() : this(null)
	    {
	        
	    }
        
        public Neuron(Type activationFunctionType)
        {
            if (activationFunctionType == null) activationFunctionType = typeof (Sigmoid);
			Id = Guid.NewGuid();
			InputSynapses = new List<Synapse>();
			OutputSynapses = new List<Synapse>();
			Initialization();
            ActivationFunction = Activator.CreateInstance(activationFunctionType) as IActivationFunction;
		}

	    public void Initialization()
	    {
	        Bias = Network.GetRandom();
	    }

	    public Neuron(IEnumerable<Neuron> inputNeurons, Type activationFunctionType = null) : this(activationFunctionType)
		{
			foreach (var inputNeuron in inputNeurons)
			{
				var synapse = new Synapse(inputNeuron, this);
				inputNeuron.OutputSynapses.Add(synapse);
				InputSynapses.Add(synapse);
			}
		}
		#endregion

		#region -- Values & Weights --
		public virtual double CalculateValue()
		{
		    var inputSignals = InputSynapses.Sum(a => a.Weight * a.InputNeuron.Value);
		    return Value = ActivationFunction.Output(inputSignals + Bias);
		}

	    public double CalculateError(double target)
		{
			return target - Value;
		}

		public double CalculateGradient(double? target = null)
		{
		    var derivative = ActivationFunction.Derivative(Value);
		    if (target != null)
            {
                Gradient = derivative*CalculateError(target.Value);
            }
		    else
            {
                Gradient = derivative*OutputSynapses.Sum(synapse => synapse.OutputNeuron.Gradient*synapse.Weight);
            }

		    return Gradient;
		}

        //public void UpdateWeights(double learnRate, double momentum)
        //{
        //    var prevDelta = BiasDelta;
        //    BiasDelta = learnRate * Gradient;
        //    Bias += BiasDelta + momentum * prevDelta;

        //    foreach (var synapse in InputSynapses)
        //    {
        //        prevDelta = synapse.WeightDelta;
        //        synapse.WeightDelta = learnRate * Gradient * synapse.InputNeuron.Value;
        //        synapse.Weight += synapse.WeightDelta + momentum * prevDelta;
        //    }
        //}
		#endregion

	    public void ResetNeuron()
	    {
	        Initialization();
	        InputSynapses.ForEach(synapse => synapse.Initialization());
	    }
	}

    public class GenericNeuron<T> : Neuron where T : IActivationFunction
    {
        public GenericNeuron() : base(typeof (T))
        {
            
        }

        public GenericNeuron(IEnumerable<Neuron> inputNeurons) : base(inputNeurons, typeof (T))
        {
            
        }
    }
}