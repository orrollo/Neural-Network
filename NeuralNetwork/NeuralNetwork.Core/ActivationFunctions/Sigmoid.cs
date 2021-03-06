﻿using System;

namespace NeuralNetwork.Core.ActivationFunctions
{
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
}