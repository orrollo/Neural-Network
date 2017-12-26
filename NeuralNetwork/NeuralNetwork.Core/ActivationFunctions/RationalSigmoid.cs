using System;

namespace NeuralNetwork.Core.ActivationFunctions
{
    public class RationalSigmoid : ActivationNumeric, IActivationFunction
    {
        public override double Output(double x)
        {
            var value = x/(Math.Abs(x) + 0.05);
            return (value + 1.0)/2.0;
        }

        protected override double OutputSolve(double y)
        {
            var y0 = 2*y - 1;
            if (Math.Abs(y0) < 1e-10) return 0.0;
            var x1 = 0.05/(1/y0 - 1);
            var x2 = 0.05/(1/y0 + 1);
            return y0 > 0 ? x1 : x2;
            return base.OutputSolve(y);
        }
    }
}