using System;

namespace NeuralNetwork.Core.ActivationFunctions
{
    public class Tanh : ActivationNumeric, IActivationFunction
    {
        public override double Output(double x)
        {
            return Math.Tanh(x);
        }

        public override double Derivative(double x)
        {
            return 1.0 - x * x;
        }
    }
}