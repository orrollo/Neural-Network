namespace NeuralNetwork.Core.TrainParams
{
    public class BackPropTrainParams : TrainParams
    {
        public TrainingType Training { get; set; }
        public double MinimumError { get; set; }
        public int NumEpochs { get; set; }

        public int ResultEpochs { get; set; }
    }
}