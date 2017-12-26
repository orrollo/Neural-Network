namespace NeuralNetwork.Core.ActivationFunctions
{
    public interface IActivationFunction
    {
        double Output(double x);
        double Derivative(double x);
    }
}