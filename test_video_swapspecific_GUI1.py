'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:00:42
Description: 
'''


###--- supress error message when detsize < 640 ---###
import onnxruntime
onnxruntime.set_default_logger_severity(3)
###################

import time
import os
import glob
import shutil
import cv2
import torch
import fractions
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import moviepy.editor as mp

from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
from tqdm import tqdm

import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd

from PIL import Image


def btnVideo():
    global video_filename
    filetypes = (
        ('Video', '*.mp4 *.avi *.gif'),
        ('All files', '*.*')
    )
    video_filename = fd.askopenfilename(title='Open Video',filetypes=filetypes)
    os.path.basename(video_filename)
    Label(tkFenster, text=os.path.basename(video_filename)).place(x=75, y=24)
    
def btnImage():
    global image_filename
    filetypes = (
        ('Images', '*.jpg *.bmp *.png'),
        ('All files', '*.*')
    )
    image_filename = fd.askopenfilename(title='Open image',filetypes=filetypes)
    os.path.basename(image_filename)
    Label(tkFenster, text=os.path.basename(image_filename)).place(x=75, y=64)

def btnOutput():
    global output_filename
    filetypes = (
        ('Video', '*.mp4'),
        ('All files', '*.*')
    )
    output_filename = fd.asksaveasfilename(defaultextension=".mp4", filetypes=filetypes)
    os.path.basename(output_filename)
    Label(tkFenster, text=os.path.basename(output_filename)).place(x=75, y=104)

def getCheckboxValue():
    checkedOrNot = cb_mask.get()

def maskClick():
    global mask
    global cb_mask
    mask = cb_mask.get()
    
def btnStart():
    global image_filename
    global video_filename
    global output_filename
    if image_filename == '' or video_filename == '' or output_filename == '':
        messagebox.showerror("Error", "Select video and image files")
    else:
        startup()
    
def btnExit():
    MsgBox = tk.messagebox.askquestion ('Exit App','Really Quit?',icon = 'error')
    if MsgBox == 'yes':
       sys.exit()

def select_crop():
    global crop_size
    crop_size = crop.get()
    
def select_detsize():
    global detsize
    detsize = det.get()

    
###--- select faces by mouse ---###

def select_specific_face(source_image):
    
    showCrosshair = False
    show_cropped = source_image

    r = cv2.selectROI("Video: Select the face to be replaced", show_cropped,showCrosshair)

    if r == (0, 0, 0, 0):
        source_image = source_image
    else:    
        source_image = source_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.destroyAllWindows()

    source_image = source_image[:, :, :3]
    return source_image
    
def select_source_face(source_image):
    
    showCrosshair = False
    show_cropped = source_image

    r = cv2.selectROI("Image: Select the face to be inserted", show_cropped,showCrosshair)

    if r == (0, 0, 0, 0):
        source_image = source_image
    else:    
        source_image = source_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.destroyAllWindows()

    source_image = source_image[:, :, :3]
    return source_image    
    
##################################

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# detransformer = transforms.Compose([
#         transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
#         transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
#     ])


###--- videoswap.py ---###

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor,specific_person_id_nonorm,id_thres, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224):
    video_forcheck = VideoFileClip(video_path)
    global use_mask
    global mask
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    #logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    frame_index = 0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #frame_count = frame_count - opt.start_pos
    
    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    mse = torch.nn.MSELoss().to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    if mask == 1:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth)) if torch.cuda.is_available() else net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu'))) if torch.cuda.is_available() else net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
        net.eval()
    else:
        net =None
    
    #video.set(1, opt.start_pos)
    # while ret:
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if  ret:
            detect_results = detect_model.get(frame,crop_size)

            if detect_results is not None:
                # print(frame_index)
                if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                frame_align_crop_list = detect_results[0]
                frame_mat_list = detect_results[1]

                id_compare_values = [] 
                frame_align_crop_tenor_list = []
                for frame_align_crop in frame_align_crop_list:

                    # BGR TO RGB
                    # frame_align_crop_RGB = frame_align_crop[...,::-1]

                    frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

                    frame_align_crop_tenor_arcnorm = spNorm(frame_align_crop_tenor)
                    frame_align_crop_tenor_arcnorm_downsample = F.interpolate(frame_align_crop_tenor_arcnorm, size=(112,112))
                    frame_align_crop_crop_id_nonorm = swap_model.netArc(frame_align_crop_tenor_arcnorm_downsample)

                    id_compare_values.append(mse(frame_align_crop_crop_id_nonorm,specific_person_id_nonorm).detach().cpu().numpy())
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                id_compare_values_array = np.array(id_compare_values)
                min_index = np.argmin(id_compare_values_array)
                min_value = id_compare_values_array[min_index]
                if min_value < id_thres:
                    swap_result = swap_model(None, frame_align_crop_tenor_list[min_index], id_vetor, None, True)[0]
                
                    #reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame,os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),pasring_model =net, norm = spNorm)
                    
                    frame = reverse2wholeimage([frame_align_crop_tenor_list[min_index]], [swap_result], [frame_mat_list[min_index]], crop_size, frame, os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),pasring_model =net, norm = spNorm)
                else:
                    if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                    frame = frame.astype(np.uint8)
                    #if not no_simswaplogo:
                    #    frame = logoclass.apply_frames(frame)
                    #cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

            else:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                #if not no_simswaplogo:
                #    frame = logoclass.apply_frames(frame)
                #cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

            cv2.imshow("Preview - Press 'Esc' to abort" ,frame)
            k = cv2.waitKey(1)
            if k == 27:  ##ord('q'):
                cv2.destroyAllWindows()
                os.system('cls')
                print ("inference aborted")
                if  os.path.exists(temp_results_dir):
                    shutil.rmtree(temp_results_dir)
                video.release()
                return
                              
            cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

        else:
            break

    video.release()

    # image_filename_list = []
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)


    clips.write_videofile(save_path,audio_codec='aac')
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)
    
###--- reverse2original.py ---###

def encode_segmentation_rgb(segmentation, no_neck=True):
    parse = segmentation

    face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    mouth_id = 11
    # hair_id = 17
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
    # hair_map = np.zeros([parse.shape[0], parse.shape[1]])

    for valid_id in face_part_ids:
        valid_index = np.where(parse==valid_id)
        face_map[valid_index] = 255
    valid_index = np.where(parse==mouth_id)
    mouth_map[valid_index] = 255
    # valid_index = np.where(parse==hair_id)
    # hair_map[valid_index] = 255
    #return np.stack([face_map, mouth_map,hair_map], axis=2)
    return np.stack([face_map, mouth_map], axis=2)


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


def postprocess(swapped_face, target, target_mask,smooth_mask):
    # target_mask = cv2.resize(target_mask, (self.size,  self.size))

    mask_tensor = torch.from_numpy(target_mask.copy().transpose((2, 0, 1))).float().mul_(1/255.0).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    face_mask_tensor = mask_tensor[0] + mask_tensor[1]
    
    soft_face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
    soft_face_mask_tensor.squeeze_()

    soft_face_mask = soft_face_mask_tensor.cpu().numpy()
    soft_face_mask = soft_face_mask[:, :, np.newaxis]

    result =  swapped_face * soft_face_mask + target * (1 - soft_face_mask)
    result = result[:,:,::-1]# .astype(np.uint8)
    return result

def reverse2wholeimage(b_align_crop_tenor_list,swaped_imgs, mats, crop_size, oriimg,  save_path = '', pasring_model = None,norm = None):

    target_image_list = []
    img_mask_list = []
    global mask
    if mask == 1:
        smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    else:
        pass

    # print(len(swaped_imgs))
    # print(mats)
    # print(len(b_align_crop_tenor_list))
    for swaped_img, mat ,source_img in zip(swaped_imgs, mats,b_align_crop_tenor_list):
        swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0))
        img_white = np.full((crop_size,crop_size), 255, dtype=float)

        # inverse the Affine transformation matrix
        mat_rev = np.zeros([2,3])
        div1 = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
        mat_rev[0][0] = mat[1][1]/div1
        mat_rev[0][1] = -mat[0][1]/div1
        mat_rev[0][2] = -(mat[0][2]*mat[1][1]-mat[0][1]*mat[1][2])/div1
        div2 = mat[0][1]*mat[1][0]-mat[0][0]*mat[1][1]
        mat_rev[1][0] = mat[1][0]/div2
        mat_rev[1][1] = -mat[0][0]/div2
        mat_rev[1][2] = -(mat[0][2]*mat[1][0]-mat[0][0]*mat[1][2])/div2

        orisize = (oriimg.shape[1], oriimg.shape[0])
        if mask == 1:
            source_img_norm = norm(source_img)
            source_img_512  = F.interpolate(source_img_norm,size=(512,512))
            out = pasring_model(source_img_512)[0]
            parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
            vis_parsing_anno = parsing.copy().astype(np.uint8)
            tgt_mask = encode_segmentation_rgb(vis_parsing_anno)
            if tgt_mask.sum() >= 5000:
                # face_mask_tensor = tgt_mask[...,0] + tgt_mask[...,1]
                target_mask = cv2.resize(tgt_mask, (crop_size,  crop_size))
                # print(source_img)
                target_image_parsing = postprocess(swaped_img, source_img[0].cpu().detach().numpy().transpose((1, 2, 0)), target_mask,smooth_mask)
                

                target_image = cv2.warpAffine(target_image_parsing, mat_rev, orisize)
                # target_image_parsing = cv2.warpAffine(swaped_img, mat_rev, orisize)
            else:
                target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)[..., ::-1]
        else:
            target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)
        # source_image   = cv2.warpAffine(source_img, mat_rev, orisize)

        img_white = cv2.warpAffine(img_white, mat_rev, orisize)


        img_white[img_white>20] =255

        img_mask = img_white

        # if use_mask:
        #     kernel = np.ones((40,40),np.uint8)
        #     img_mask = cv2.erode(img_mask,kernel,iterations = 1)
        # else:
        kernel = np.ones((40,40),np.uint8)
        img_mask = cv2.erode(img_mask,kernel,iterations = 1)
        kernel_size = (20, 20)
        blur_size = tuple(2*i+1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

        # kernel = np.ones((10,10),np.uint8)
        # img_mask = cv2.erode(img_mask,kernel,iterations = 1)



        img_mask /= 255

        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])

        # pasing mask

        # target_image_parsing = postprocess(target_image, source_image, tgt_mask)

        if mask == 1:
            target_image = np.array(target_image, dtype=np.float) * 255
        else:
            target_image = np.array(target_image, dtype=np.float)[..., ::-1] * 255


        img_mask_list.append(img_mask)
        target_image_list.append(target_image)
        

    # target_image /= 255
    # target_image = 0
    img = np.array(oriimg, dtype=np.float)
    for img_mask, target_image in zip(img_mask_list, target_image_list):
        img = img_mask * target_image + (1-img_mask) * img
        
    final_img = img.astype(np.uint8)

    #cv2.imwrite(save_path, final_img)
    
    
    #------- preview ------#
    #(h, w) = final_img.shape[:2]
    #h = h//2
    #w = w//2
    #final_img = cv2.resize(final_img, (w, h))
    ##cv2.imshow("Final Video", final_img)
    ##cv2.waitKey(1)
    return final_img

    
###--- Start inference ---###
   
def startup():
    tkFenster.withdraw()
    global crop_size
    global detsize
    global cb_mask

    opt = TestOptions().parse()
    opt.Arc_path = 'arcface_model/arcface_checkpoint.tar'
    opt.pic_a_path = image_filename
    opt.video_path = video_filename
    opt.output_path = output_filename
    opt.crop_size = crop_size

    #pic_specific = opt.pic_specific_path
    start_epoch, epoch_iter = 1, 0
    
    if mask == 1:
        use_mask = True
    else:
        use_mask = False    

    torch.nn.Module.dump_patches = True
    
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
        
    model = create_model(opt)
    model.eval()

    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(detsize, detsize),mode='none')
    with torch.no_grad():
        pic_a = image_filename # opt.pic_a_path
        #img_a = Image.open(pic_a).convert('RGB')

        img_a_whole = cv2.imread(pic_a)
        img_a_whole = select_source_face(img_a_whole)
        cv2.destroyAllWindows()
        
        
        img_a_align_crop, _ = app.get(img_a_whole,crop_size)

        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])


        # convert numpy to tensor
        img_id = img_id.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        # img_att = img_att.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)
        
###--- select face to be swapped by mouse ---###

        os.system('cls')
        video = cv2.VideoCapture(opt.video_path)
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print ("Number of frames: " + str(n_frames))
        
        # enter videoposition for face selection manually
        
        try:
            videoposition = float(input("Enter frame number to select the face to be swapped: "))
        except ValueError:
            videoposition = 1
        
        video.set(1,videoposition)
        ret, image = video.read()
        
        specific_person_whole = select_specific_face(image)
        
        video.release()
        cv2.destroyAllWindows()
        os.system('cls')
        

        # The specific person to be swapped
        #specific_person_whole = cv2.imread(pic_specific)
        temp_results_dir=opt.temp_path
        
        #try:
        
        specific_person_align_crop, _ = app.get(specific_person_whole,crop_size)
        specific_person_align_crop_pil = Image.fromarray(cv2.cvtColor(specific_person_align_crop[0],cv2.COLOR_BGR2RGB)) 
        specific_person = transformer_Arcface(specific_person_align_crop_pil)
        specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1], specific_person.shape[2])
        specific_person = specific_person.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        specific_person_downsample = F.interpolate(specific_person, size=(112,112))
        specific_person_id_nonorm = model.netArc(specific_person_downsample)
            
        print ("Swapping face...")
        print ("")
        print ("Video: " + opt.video_path)
        print ("Image: " + opt.pic_a_path)
        print ("Output: " + opt.output_path)
        print ("Crop size: " + str(opt.crop_size))
        print ("Detection size: " + str(detsize))
        print ("Use mask: " + str(use_mask))
        print ("")
        
        #try:
        
        video_swap(opt.video_path, latend_id,specific_person_id_nonorm, opt.id_thres, model, app, opt.output_path, temp_results_dir, crop_size)
        cv2.destroyAllWindows()
        tkFenster.deiconify()

        #except:    
            #print ("Error ! Maybe no face in image or video detected. Choose other files")
            #tkFenster.deiconify()
            
###--- GUI ---###

tkFenster = Tk()
tkFenster.title('SimSwapSpecific 0.1')
tkFenster.geometry('320x240')

crop = tk.IntVar()
crop.set (224)

det = tk.IntVar()
det.set (640)

cb_mask = tk.IntVar()
cb_mask.set(0)
use_mask = False
mask = 0   

image_filename = ''
video_filename = ''
output_filename = ''
crop_size = 224
detsize = 640

Button(tkFenster, text=' Video  ', command=btnVideo).place(x=20, y=20)
Button(tkFenster, text=' Image ', command=btnImage).place(x=20, y=60)
Button(tkFenster, text='  Out    ',  command=btnOutput).place(x=20, y=100)
Button(tkFenster, text='  Start  ',  command=btnStart).place(x=20, y=210)
Button(tkFenster, text='  Exit  ',  command=btnExit).place(x=80, y=210)

Label(tkFenster, text=' - ').place(x=75, y=24)
Label(tkFenster, text=' - ').place(x=75, y=64)
Label(tkFenster, text=' - ').place(x=75, y=104)
Label(tkFenster, text='Cropsize:').place(x=20, y=140)
Label(tkFenster, text='Detsize:').place(x=20, y=160)

radiobutton1 = Radiobutton(master=tkFenster, anchor='w',text='224', value='224',command=select_crop, variable=crop)
radiobutton1.place(x=80, y=140, width=60, height=20)
radiobutton2 = Radiobutton(master=tkFenster, anchor='w',text='512', value='512',command=select_crop, variable=crop)
radiobutton2.place(x=130, y=140, width=80, height=20)

radiobutton3 = Radiobutton(master=tkFenster, anchor='w',text='256', value='256',command=select_detsize, variable=det)
radiobutton3.place(x=80, y=160, width=60, height=20)
radiobutton4 = Radiobutton(master=tkFenster, anchor='w',text='320', value='320',command=select_detsize, variable=det)
radiobutton4.place(x=130, y=160, width=60, height=20)
radiobutton5 = Radiobutton(master=tkFenster, anchor='w',text='480', value='480',command=select_detsize, variable=det)
radiobutton5.place(x=180, y=160, width=60, height=20)
radiobutton6 = Radiobutton(master=tkFenster, anchor='w',text='640', value='640',command=select_detsize, variable=det)
radiobutton6.place(x=230, y=160, width=60, height=20)
radiobutton1.select()

checkbutton0 = Checkbutton(master=tkFenster, anchor='w',text='Use Mask', offvalue=0, onvalue=1, variable=cb_mask,command=maskClick)
checkbutton0.place(x=20, y=180, width=100, height=20)

tkFenster.mainloop()

