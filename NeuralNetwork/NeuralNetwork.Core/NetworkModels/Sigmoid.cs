using System;

namespace NeuralNetwork.NetworkModels
{
    public interface IActivationFunction
    {
        double Output(double x);
        double Derivative(double x);
    }

    public class Sigmoid : IActivationFunction
    {
        public double Output(double x)
        {
            return x < -45.0 ? 0.0 : x > 45.0 ? 1.0 : 1.0 / (1.0 + Math.Exp(-x));
        }

        public double Derivative(double x)
        {
            return x * (1 - x);
        }
    }

    //public static class XSigmoid
    //{
    //    public static double Output(double x)
    //    {
    //        return x < -45.0 ? 0.0 : x > 45.0 ? 1.0 : 1.0 / (1.0 + Math.Exp(-x));
    //    }

    //    public static double Derivative(double x)
    //    {
    //        return x * (1 - x);
    //    }
    //}
}