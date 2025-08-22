class poseidonModelWrapper:
    def __init__(self, pc_encoder, rgb_encoder=None, shared_metric_space=None, pose_estimator=None, contrastive_loss=None, grad_norm=None):
        self.pc_encoder = pc_encoder
        self.rgb_encoder = rgb_encoder
        self.shared_metric_space = shared_metric_space
        self.pose_estimator = pose_estimator
        self.contrastive_loss = contrastive_loss
        self.grad_norm = grad_norm
        self.max_grad_norm = 1.0  # Add gradient clipping threshold

    def get_trainable_modules(self):
        modules = {
            "pc_encoder": {
                "model": self.pc_encoder, 
                "params": {
                    "learning_rate": 0.0001,  # Reduced learning rate
                    "weight_decay": 1e-4,     # Increased weight decay
                    "scheduler_step_size": 3,  # More frequent updates
                    "scheduler_gamma": 0.95    # Gentler decay
                }
            },
            "pose_estimator": {
                "model": self.pose_estimator, 
                "params": {
                    "learning_rate": 0.0001,  # Reduced learning rate
                    "weight_decay": 1e-4,     # Increased weight decay
                    "scheduler_step_size": 3,  # More frequent updates
                    "scheduler_gamma": 0.95    # Gentler decay
                }
            },
        }
        if self.rgb_encoder:
            modules["rgb_encoder"] = {
                "model": self.rgb_encoder, 
                "params": {
                    "learning_rate": 1e-5,
                    "weight_decay": 1e-4,
                    "scheduler_step_size": 3,
                    "scheduler_gamma": 0.95
                }
            }
        if self.shared_metric_space:
            modules["shared_metric_space"] = {
                "model": self.shared_metric_space, 
                "params": {
                    "learning_rate": 1e-5,
                    "weight_decay": 1e-4,
                    "scheduler_step_size": 3,
                    "scheduler_gamma": 0.95
                }
            }
        #we handle training of grad_norm directly in the training loop
        return modules
    
    def to_device(self, device):
        self.pc_encoder.to(device)
        if self.rgb_encoder:
            self.rgb_encoder.to(device)
        if self.shared_metric_space:
            self.shared_metric_space.to(device)
        self.pose_estimator.to(device)

    def activate_training_mode(self):
        self.pc_encoder.train()
        self.pose_estimator.train()
        #if self.rgb_encoder:
        #    self.rgb_encoder.train()
        if self.shared_metric_space:
            self.shared_metric_space.train()
        if self.grad_norm:
            self.grad_norm.train()
    
    def activate_eval_mode(self):
        self.pc_encoder.eval()
        self.pose_estimator.eval()
        if self.rgb_encoder:
            self.rgb_encoder.eval()
        if self.shared_metric_space:
            self.shared_metric_space.eval()
        if self.grad_norm:
            self.grad_norm.eval()

    def get_checkpoint(self):
        checkpoint = {
            'pc_encoder_state_dict': self.pc_encoder.state_dict(),
            'pose_estimator_state_dict': self.pose_estimator.state_dict()
        }
        if self.rgb_encoder:
            checkpoint['rgb_encoder_state_dict'] = self.rgb_encoder.state_dict()
        if self.shared_metric_space:
            checkpoint['shared_metric_space'] = self.shared_metric_space.state_dict()
        
        return checkpoint
    
    def load_checkpoint(self, checkpoint):
        self.pc_encoder.load_state_dict(checkpoint['pc_encoder_state_dict'])
        self.pose_estimator.load_state_dict(checkpoint['pose_estimator_state_dict'])
        if self.rgb_encoder:
            self.rgb_encoder.load_state_dict(checkpoint['rgb_encoder_state_dict'])
        if self.shared_metric_space:
            self.shared_metric_space.load_state_dict(checkpoint['shared_metric_space'])
        

