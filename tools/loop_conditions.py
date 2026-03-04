def to_log(step, total_len, log_at_these_steps):
    return (step % log_at_these_steps == 0) or ((step + 1) == total_len)

def to_save_checkpoint(epoch, total_epochs, log_at_these_epochs, to_checkpoint=True):
    if not to_checkpoint:
        return False
    return (epoch % log_at_these_epochs == 0) or ((epoch + 1) == total_epochs)

def to_visualize_images(step, total_steps, log_at_these_steps):
    return (step % log_at_these_steps == 0)  or ((step + 1) == total_steps)

def to_visualize_images_epoch(epoch, total_epochs, log_at_these_epochs):
    return (epoch % log_at_these_epochs == 0)  or ((epoch + 1) == total_epochs)