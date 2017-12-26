using System;

namespace NeuralNetwork.Core.ActivationFunctions
{
    public abstract class ActivationNumeric
    {
        public virtual double Derivative(double y)
        {
            var x = OutputSolve(y);
            return adaptDiff(x);
        }

        private double adaptDiff(double x)
        {
            var h0 = 0.5;
            double der1 = intDiff(x, h0);
            while (true)
            {
                double der2 = intDiff(x, h0/2);
                double e2 = Math.Abs(der1 - der2)/15.0;
                der1 = der2;
                if (e2 < 1e-4) break;
                h0 = h0/2;
            }
            return der1;
        }

        private double intDiff(double x, double h)
        {
            return (Output(x - 2*h) - 8*Output(x - h) + 8*Output(x + h) - Output(x + 2*h))/(12*h);
        }

        protected virtual double OutputSolve(double y)

        {
            double left = -0.5, right = 0.5;

            while (Output(left) > y) { right = left; left = left*4/3; }
            while (Output(right) < y) { left = right; right = right*4/3; }

            while ((right - left) > 1e-5)
            {
                var mid = (left + right)/2.0;
                var fn = Output(mid) - y;
                if (Math.Abs(fn) < 1e-5) return mid;
                if (fn > 0)
                    right = mid;
                else
                    left = mid;
            }

            return (left + right)/2.0;
        }

        public abstract double Output(double x);
    }
}