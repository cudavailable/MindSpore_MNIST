class TrainInfo:
    """ contain training information """
    def __init__(self, model, loss_fn, optimizer, grad_fn, data=None, logger=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_fn = grad_fn
        self.data = data
        self.logger = logger

def train_step(data, label, train_info):
    """ train in 1 batch """
    (loss, _), grad = train_info.grad_fn(data, label, train_info.loss_fn, train_info.model)
    train_info.optimizer(grad)
    return loss

def train(train_info):
    """ train in 1 epoch """
    dataset = train_info.data # get training datasets
    model = train_info.model # get model
    logger = train_info.logger # get logger

    total_size = dataset.get_dataset_size()
    model.set_train()  # set training mode
    for batch, (image, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(image, label, train_info)

        if batch % 100 == 0:
            loss = loss.asnumpy()
            logger.write(f"[{batch}/{total_size}]  loss : {loss:0.6f} \n")


def test(test_info):
    """ test in 1 epoch """
    # initialize test info
    dataset = test_info.data
    model = test_info.model
    loss_fn = test_info.loss_fn
    logger = test_info.logger

    total_size = dataset.get_dataset_size()
    model.set_train(False)
    total, correct, test_loss = 0, 0, 0
    for image, label in dataset.create_tuple_iterator():
        pred = model(image)
        total += len(image)
        correct += (pred.argmax(1) == label).asnumpy().sum()
        test_loss += loss_fn(pred, label).asnumpy()

    test_loss /= total_size
    correct /= total
    logger.write(f"Test: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f} \n\n")

    return test_loss, correct