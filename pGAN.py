def train():
    import time
    from options.train_options import TrainOptions
    from data.npy_data_processor import ImageDataset
    from torch.utils.data import DataLoader
    from models import create_model
    from util.visualizer import Visualizer

    opt = TrainOptions().parse()
    opt.input_nc = len(opt.model_xs)
    model = create_model(opt)
    # Loading data
    dataset = ImageDataset(opt)
    dataset_size = len(dataset)
    print('Training images = %d' % dataset_size)
    data_loader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=opt.serial_batches, num_workers=opt.nThreads)

    visualizer = Visualizer(opt)
    total_steps = 0
    # Starts training
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            # Save current images (real_A, real_B, fake_B)
            if epoch_iter % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, epoch_iter, save_result)
            # Save current errors
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)
            # Save model based on the number of iterations
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()
        # Save model based on the number of epochs
        print(opt.dataset_mode)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


def test():
    import sys
    sys.argv = args
    import os
    from options.test_options import TestOptions
    from data.npy_data_processor import ImageDataset
    from torch.utils.data import DataLoader
    from models import create_model
    import numpy as np

    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.input_nc = len(opt.model_xs)

    dataset = ImageDataset(opt)
    dataset_size = len(dataset)
    print('Training images = %d' % dataset_size)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.nThreads)

    model = create_model(opt)
    # test
    for i, data in enumerate(data_loader):
        # if i >= opt.how_many:
        #     break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()[0]
        patients_path = os.path.split(os.path.split(os.path.split(img_path)[0])[0])[-1]
        os.makedirs(os.path.join(opt.results_dir, opt.name, patients_path), exist_ok=True)

        slicer_id = os.path.split(img_path)[-1]
        np.save(os.path.join(opt.results_dir, opt.name, patients_path, slicer_id),
                visuals['fake_B'].detach().cpu().numpy())
        print('process image... %s' % img_path)


import sys

if __name__ == '__main__':

    # sys.argv.extend(['--model', 'pGAN', '--training'])
    sys.argv.extend(['--model', 'pGAN'])

    args = sys.argv
    if '--training' in str(args):
        train()
    else:
        sys.argv.extend(['--serial_batches'])
        test()
