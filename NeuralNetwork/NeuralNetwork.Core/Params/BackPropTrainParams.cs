namespace NeuralNetwork.Core.Params
{
    public class BackPropTrainParams : TrainParams
    {
        private int _numEpochs = int.MaxValue;
        public TrainingType Training { get; set; }
        public double MinimumError { get; set; }

        public int NumEpochs
        {
            get { return _numEpochs; }
            set { _numEpochs = value; }
        }

        public double LearnRate { get; set; }
        public double Momentum { get; set; }

        public int ResultEpochs { get; set; }
    }
}