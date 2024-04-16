def start_train(data,batch,train_model,max_train_epochs,img_size,device,output_name,worker,out_dir):
    if st.session_state.clicked:
        st.warning("训练正在进行中，请等待当前训练完成。")
        return

    # 将按钮点击状态设置为已点击
    click_button()

    try:
        # 执行训练任务
        st.session_state.clicked = True
        model = YOLO(train_model)
        result = model.train(data=data, batch=batch,epochs=max_train_epochs, imgsz=img_size, device=device,  name=output_name,workers=worker, save_dir=out_dir)
       
        return result
    except Exception as e:
        traceback.print_exc()
        st.error('运行出现错误：{}'.format(str(e)))
        
    finally:
        # 训练完成后，将程序状态设置为未执行
        st.session_state.clicked = False
##抛错加防止重复标准 
 
 


 
 time	None	最长训练时间（小时）。如果设置了该值，则会覆盖 epochs 参数，允许训练在指定的持续时间后自动停止。对于时间有限的训练场景非常有用。
 cache	False	在内存中缓存数据集图像 (True/ram）、磁盘 (disk），或禁用它 (False).通过减少磁盘 I/O 提高训练速度，但代价是增加内存使用量。
 rect	False	可进行矩形训练，优化批次组成以减少填充。这可以提高效率和速度，但可能会影响模型的准确性。
 cos_lr	False	利用余弦学习率调度器，根据历时的余弦曲线调整学习率。这有助于管理学习率，实现更好的收敛。
 lrf	0.01	最终学习率占初始学习率的百分比 = (lr0 * lrf)，与调度程序结合使用，随着时间的推移调整学习率。
 lr0	0.01	初始学习率（即 SGD=1E-2, Adam=1E-3) .调整这个值对优化过程至关重要，会影响模型权重的更新速度。
 amp	True	启用自动混合精度 (AMP) 训练，可减少内存使用量并加快训练速度，同时将对精度的影响降至最低。


 {'settings_version': '0.0.4', 'datasets_dir': 'D:\\IDE\\vscode\\streamlit\\yolo_trainner\\datasets', 'weights_dir': 'weights', 'runs_dir': 'runs', 'uuid': '24b76980ea0fa7b0bd793a98c284077bdb6e92addf464c0cb54128ab1bb9f4ce', 'sync': True, 'api_key': '', 'openai_api_key': '', 'clearml': True, 'comet': True, 'dvc': True, 'hub': True, 'mlflow': True, 'neptune': True, 'raytune': True, 'tensorboard': True, 'wandb': True}

 
 task=detect, 

 mode=train, 
 model=./weights\detection\yolov8n.pt,
  data=./datasets/coco128.yaml, 
  epochs=5, 
  time=None,
   patience=100,
    batch=1, 
    imgsz=640, 
    save=True,
     save_period=-1, 
     cache=False,
      device=[0],
       workers=0,
        project=None, 
        name=demo3,
         exist_ok=False,
          pretrained=True,
           optimizer=auto,
            verbose=True,
             seed=0,
              deterministic=True,
               single_cls=False,
                rect=False,
                 cos_lr=False,
                  close_mosaic=10,
                   resume=False,
                    amp=True,
                     fraction=1.0,
                      profile=False,
                       freeze=None,
                        multi_scale=False,
                         overlap_mask=True,
                          mask_ratio=4,
                           dropout=0.0, 
                           val=True,
                            split=val,
                             save_json=False, 
                             save_hybrid=False,
                              conf=None, iou=0.7,
                               max_det=300,
                                half=False,
                                 dnn=False, 
                                 plots=True, 
                                 source=None,
                                  vid_stride=1,
                                   stream_buffer=False,
                                    visualize=False,
                                     augment=False,
                                      agnostic_nms=False,
                                       classes=None,
                                        retina_masks=False, 
                                        embed=None, 
                                        show=False, 
                                        save_frames=False,
                                         save_txt=False, 
                                         save_conf=False,
                                          save_crop=False, 
                                          show_labels=True,
                                           show_conf=True,
                                            show_boxes=True,
                                             line_width=None, format=torchscript,
                                              keras=False,
                                               optimize=False,
                                                int8=False,
                                                 dynamic=False,
                                                  simplify=False,
                                                   opset=None,
                                                    workspace=4,
                                                     nms=False,
                                                      lr0=0.01, 
                                                      lrf=0.01,
                                                       momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs\detect\demo3