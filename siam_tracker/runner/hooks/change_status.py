from .hook import Hook


class ChangeStatusHook(Hook):

    def __init__(self,
                 action_list):
        # [(20, 'with_meta_train', True)]
        # [epoch, key, value]
        self.action_list = action_list

    def before_run(self, runner):
        for last_epoch in range(runner.epoch):
            for epoch, key, value in self.action_list:
                if last_epoch == epoch:
                    if hasattr(runner.model, 'module'):
                        m = runner.model.module
                    else:
                        m = runner.model
                    runner.logger.info('Set model.{} --> {} (Epoch {})'.format(key, value, epoch))
                    self.set(m, key, value)

    def before_epoch(self, runner):
        for epoch, key, value in self.action_list:
            if runner.epoch == epoch:
                if hasattr(runner.model, 'module'):
                    m = runner.model.module
                else:
                    m = runner.model
                runner.logger.info('Set model.{} --> {} (Epoch {})'.format(key, value, epoch))
                self.set(m, key, value)

    @staticmethod
    def set(model, key, value):
        sps = key.split('.')
        root = model
        for i in range(len(sps)):
            if hasattr(root, sps[i]):
                if i == len(sps) - 1:
                    setattr(root, sps[i], value)
                else:
                    root = getattr(root, sps[i])
            else:
                print("Cannot find {}".format(key))
