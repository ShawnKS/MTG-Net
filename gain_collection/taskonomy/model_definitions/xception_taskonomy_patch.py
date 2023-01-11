--- taskgrouping/model_definitions/xception_taskonomy_old.py	2021-06-02 12:28:53.000000000 -0700
+++ taskgrouping/model_definitions/xception_taskonomy_new.py	2021-06-02 12:14:39.000000000 -0700
@@ -6,9 +6,15 @@
 import torch.utils.model_zoo as model_zoo
 from torch.nn import init
 import torch
-from .ozan_rep_fun import ozan_rep_function,trevor_rep_function,OzanRepFunction,TrevorRepFunction
+from .ozan_rep_fun import ozan_rep_function, trevor_rep_function, OzanRepFunction, TrevorRepFunction, gradnorm_rep_function, GradNormRepFunction
+from absl import logging
 
-__all__ = ['xception_taskonomy_new','xception_taskonomy_new_fifth','xception_taskonomy_new_quad','xception_taskonomy_new_half','xception_taskonomy_new_80','xception_taskonomy_ozan']
+
+__all__ = [
+    'xception_taskonomy_new', 'xception_taskonomy_new_gradnorm', 'xception_taskonomy_new_fifth',
+    'xception_taskonomy_new_quad', 'xception_taskonomy_new_half',
+    'xception_taskonomy_new_80', 'xception_taskonomy_ozan'
+]
 
 # model_urls = {
 #     'xception_taskonomy':'file:///home/tstand/Dropbox/taskonomy/xception_taskonomy-a4b32ef7.pth.tar'
@@ -16,424 +22,515 @@
 
 
 class SeparableConv2d(nn.Module):
-    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False,groupsize=1):
-        super(SeparableConv2d,self).__init__()
 
-        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=max(1,in_channels//groupsize),bias=bias)
-        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
-        #self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias)
-        #self.pointwise=lambda x:x
-
-    def forward(self,x):
-        x = self.conv1(x)
-        x = self.pointwise(x)
-        return x
+  def __init__(self,
+               in_channels,
+               out_channels,
+               kernel_size=1,
+               stride=1,
+               padding=0,
+               dilation=1,
+               bias=False,
+               groupsize=1):
+    super(SeparableConv2d, self).__init__()
+
+    self.conv1 = nn.Conv2d(
+        in_channels,
+        in_channels,
+        kernel_size,
+        stride,
+        padding,
+        dilation,
+        groups=max(1, in_channels // groupsize),
+        bias=bias)
+    self.pointwise = nn.Conv2d(
+        in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
+    #self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias)
+    #self.pointwise=lambda x:x
+
+  def forward(self, x):
+    x = self.conv1(x)
+    x = self.pointwise(x)
+    return x
 
 
 class Block(nn.Module):
-    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
-        super(Block, self).__init__()
-
-        if out_filters != in_filters or strides!=1:
-            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
-            self.skipbn = nn.BatchNorm2d(out_filters)
-        else:
-            self.skip=None
-
-        self.relu = nn.ReLU(inplace=True)
-        rep=[]
 
-        filters=in_filters
-        if grow_first:
-            rep.append(self.relu)
-            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
-            rep.append(nn.BatchNorm2d(out_filters))
-            filters = out_filters
-
-        for i in range(reps-1):
-            rep.append(self.relu)
-            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
-            rep.append(nn.BatchNorm2d(filters))
-
-        if not grow_first:
-            rep.append(self.relu)
-            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
-            rep.append(nn.BatchNorm2d(out_filters))
-            filters=out_filters
+  def __init__(self,
+               in_filters,
+               out_filters,
+               reps,
+               strides=1,
+               start_with_relu=True,
+               grow_first=True):
+    super(Block, self).__init__()
+
+    if out_filters != in_filters or strides != 1:
+      self.skip = nn.Conv2d(
+          in_filters, out_filters, 1, stride=strides, bias=False)
+      self.skipbn = nn.BatchNorm2d(out_filters)
+    else:
+      self.skip = None
+
+    self.relu = nn.ReLU(inplace=True)
+    rep = []
+
+    filters = in_filters
+    if grow_first:
+      rep.append(self.relu)
+      rep.append(
+          SeparableConv2d(
+              in_filters, out_filters, 3, stride=1, padding=1, bias=False))
+      rep.append(nn.BatchNorm2d(out_filters))
+      filters = out_filters
+
+    for i in range(reps - 1):
+      rep.append(self.relu)
+      rep.append(
+          SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
+      rep.append(nn.BatchNorm2d(filters))
+
+    if not grow_first:
+      rep.append(self.relu)
+      rep.append(
+          SeparableConv2d(
+              in_filters, out_filters, 3, stride=1, padding=1, bias=False))
+      rep.append(nn.BatchNorm2d(out_filters))
+      filters = out_filters
+
+    if not start_with_relu:
+      rep = rep[1:]
+    else:
+      rep[0] = nn.ReLU(inplace=False)
+
+    if strides != 1:
+      #rep.append(nn.AvgPool2d(3,strides,1))
+      rep.append(nn.Conv2d(filters, filters, 2, 2))
+    self.rep = nn.Sequential(*rep)
+
+  def forward(self, inp):
+    x = self.rep(inp)
+
+    if self.skip is not None:
+      skip = self.skip(inp)
+      skip = self.skipbn(skip)
+    else:
+      skip = inp
+    x += skip
+    return x
 
-        if not start_with_relu:
-            rep = rep[1:]
-        else:
-            rep[0] = nn.ReLU(inplace=False)
-
-        if strides != 1:
-            #rep.append(nn.AvgPool2d(3,strides,1))
-            rep.append(nn.Conv2d(filters,filters,2,2))
-        self.rep = nn.Sequential(*rep)
-
-    def forward(self,inp):
-        x = self.rep(inp)
-
-        if self.skip is not None:
-            skip = self.skip(inp)
-            skip = self.skipbn(skip)
-        else:
-            skip = inp
-        x+=skip
-        return x
 
 class Encoder(nn.Module):
-    def __init__(self, sizes=[32,64,128,256,728,728,728,728,728,728,728,728,728]):
-        super(Encoder, self).__init__()
-        self.conv1 = nn.Conv2d(3, sizes[0], 3,2, 1, bias=False)
-        self.bn1 = nn.BatchNorm2d(sizes[0])
-        self.relu = nn.ReLU(inplace=True)
-        self.relu2 = nn.ReLU(inplace=False)
-
-        self.conv2 = nn.Conv2d(sizes[0],sizes[1],3,1,1,bias=False)
-        self.bn2 = nn.BatchNorm2d(sizes[1])
-        #do relu here
-
-        self.block1=Block(sizes[1],sizes[2],2,2,start_with_relu=False,grow_first=True)
-        self.block2=Block(sizes[2],sizes[3],2,2,start_with_relu=True,grow_first=True)
-        self.block3=Block(sizes[3],sizes[4],2,2,start_with_relu=True,grow_first=True)
-
-        self.block4=Block(sizes[4],sizes[5],3,1,start_with_relu=True,grow_first=True)
-        self.block5=Block(sizes[5],sizes[6],3,1,start_with_relu=True,grow_first=True)
-        self.block6=Block(sizes[6],sizes[7],3,1,start_with_relu=True,grow_first=True)
-        self.block7=Block(sizes[7],sizes[8],3,1,start_with_relu=True,grow_first=True)
-
-        self.block8=Block(sizes[8],sizes[9],3,1,start_with_relu=True,grow_first=True)
-        self.block9=Block(sizes[9],sizes[10],3,1,start_with_relu=True,grow_first=True)
-        self.block10=Block(sizes[10],sizes[11],3,1,start_with_relu=True,grow_first=True)
-        self.block11=Block(sizes[11],sizes[12],3,1,start_with_relu=True,grow_first=True)
-
-        #self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)
-
-        #self.conv3 = SeparableConv2d(768,512,3,1,1)
-        #self.bn3 = nn.BatchNorm2d(512)
-        #self.conv3 = SeparableConv2d(1024,1536,3,1,1)
-        #self.bn3 = nn.BatchNorm2d(1536)
-
-        #do relu here
-        #self.conv4 = SeparableConv2d(1536,2048,3,1,1)
-        #self.bn4 = nn.BatchNorm2d(2048)
-    def forward(self,input):
-
-        x = self.conv1(input)
-        x = self.bn1(x)
-        x = self.relu(x)
-
-        x = self.conv2(x)
-        x = self.bn2(x)
-        x = self.relu(x)
-
-        x = self.block1(x)
-        x = self.block2(x)
-        x = self.block3(x)
-        x = self.block4(x)
-        x = self.block5(x)
-        x = self.block6(x)
-        x = self.block7(x)
-        x = self.block8(x)
-        x = self.block9(x)
-        x = self.block10(x)
-        x = self.block11(x)
-        #x = self.block12(x)
-
-        #x = self.conv3(x)
-        #x = self.bn3(x)
-        #x = self.relu(x)
-
-
-        #x = self.conv4(x)
-        #x = self.bn4(x)
-
-        representation = self.relu2(x)
-
-        return representation
-
-
-
-def interpolate(inp,size):
-    t = inp.type()
-    inp = inp.float()
-    out = nn.functional.interpolate(inp,size=size,mode='bilinear',align_corners=False)
-    if out.type()!=t:
-        out = out.half()
-    return out
 
+  def __init__(
+      self,
+      sizes=[32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]):
+    super(Encoder, self).__init__()
+    self.conv1 = nn.Conv2d(3, sizes[0], 3, 2, 1, bias=False)
+    self.bn1 = nn.BatchNorm2d(sizes[0])
+    self.relu = nn.ReLU(inplace=True)
+    self.relu2 = nn.ReLU(inplace=False)
+
+    self.conv2 = nn.Conv2d(sizes[0], sizes[1], 3, 1, 1, bias=False)
+    self.bn2 = nn.BatchNorm2d(sizes[1])
+    #do relu here
+
+    self.block1 = Block(
+        sizes[1], sizes[2], 2, 2, start_with_relu=False, grow_first=True)
+    self.block2 = Block(
+        sizes[2], sizes[3], 2, 2, start_with_relu=True, grow_first=True)
+    self.block3 = Block(
+        sizes[3], sizes[4], 2, 2, start_with_relu=True, grow_first=True)
+
+    self.block4 = Block(
+        sizes[4], sizes[5], 3, 1, start_with_relu=True, grow_first=True)
+    self.block5 = Block(
+        sizes[5], sizes[6], 3, 1, start_with_relu=True, grow_first=True)
+    self.block6 = Block(
+        sizes[6], sizes[7], 3, 1, start_with_relu=True, grow_first=True)
+    self.block7 = Block(
+        sizes[7], sizes[8], 3, 1, start_with_relu=True, grow_first=True)
+
+    self.block8 = Block(
+        sizes[8], sizes[9], 3, 1, start_with_relu=True, grow_first=True)
+    self.block9 = Block(
+        sizes[9], sizes[10], 3, 1, start_with_relu=True, grow_first=True)
+    self.block10 = Block(
+        sizes[10], sizes[11], 3, 1, start_with_relu=True, grow_first=True)
+    self.block11 = Block(
+        sizes[11], sizes[12], 3, 1, start_with_relu=True, grow_first=True)
+
+    #self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)
+
+    #self.conv3 = SeparableConv2d(768,512,3,1,1)
+    #self.bn3 = nn.BatchNorm2d(512)
+    #self.conv3 = SeparableConv2d(1024,1536,3,1,1)
+    #self.bn3 = nn.BatchNorm2d(1536)
+
+    #do relu here
+    #self.conv4 = SeparableConv2d(1536,2048,3,1,1)
+    #self.bn4 = nn.BatchNorm2d(2048)
+  def forward(self, input):
+
+    x = self.conv1(input)
+    x = self.bn1(x)
+    x = self.relu(x)
+
+    x = self.conv2(x)
+    x = self.bn2(x)
+    x = self.relu(x)
+
+    x = self.block1(x)
+    x = self.block2(x)
+    x = self.block3(x)
+    x = self.block4(x)
+    x = self.block5(x)
+    x = self.block6(x)
+    x = self.block7(x)
+    x = self.block8(x)
+    x = self.block9(x)
+    x = self.block10(x)
+    x = self.block11(x)
+    #x = self.block12(x)
+
+    #x = self.conv3(x)
+    #x = self.bn3(x)
+    #x = self.relu(x)
+
+    #x = self.conv4(x)
+    #x = self.bn4(x)
+
+    representation = self.relu2(x)
+
+    return representation
+
+
+def interpolate(inp, size):
+  t = inp.type()
+  inp = inp.float()
+  out = nn.functional.interpolate(
+      inp, size=size, mode='bilinear', align_corners=False)
+  if out.type() != t:
+    out = out.half()
+  return out
 
 
 class Decoder(nn.Module):
-    def __init__(self, output_channels=32,num_classes=None,half_sized_output=False,small_decoder=True):
-        super(Decoder, self).__init__()
 
-        self.output_channels = output_channels
-        self.num_classes = num_classes
-        self.half_sized_output=half_sized_output
-        self.relu = nn.ReLU(inplace=True)
-        if num_classes is not None:
-            self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)
-
-            self.conv3 = SeparableConv2d(1024,1536,3,1,1)
-            self.bn3 = nn.BatchNorm2d(1536)
-
-            #do relu here
-            self.conv4 = SeparableConv2d(1536,2048,3,1,1)
-            self.bn4 = nn.BatchNorm2d(2048)
-
-            self.fc = nn.Linear(2048, num_classes)
+  def __init__(self,
+               output_channels=32,
+               num_classes=None,
+               half_sized_output=False,
+               small_decoder=True):
+    super(Decoder, self).__init__()
+
+    self.output_channels = output_channels
+    self.num_classes = num_classes
+    self.half_sized_output = half_sized_output
+    self.relu = nn.ReLU(inplace=True)
+    if num_classes is not None:
+      self.block12 = Block(
+          728, 1024, 2, 2, start_with_relu=True, grow_first=False)
+
+      self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
+      self.bn3 = nn.BatchNorm2d(1536)
+
+      #do relu here
+      self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
+      self.bn4 = nn.BatchNorm2d(2048)
+
+      self.fc = nn.Linear(2048, num_classes)
+    else:
+      if small_decoder:
+        self.upconv1 = nn.ConvTranspose2d(512, 128, 2, 2)
+        self.bn_upconv1 = nn.BatchNorm2d(128)
+        self.conv_decode1 = nn.Conv2d(128, 128, 3, padding=1)
+        self.bn_decode1 = nn.BatchNorm2d(128)
+        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)
+        self.bn_upconv2 = nn.BatchNorm2d(64)
+        self.conv_decode2 = nn.Conv2d(64, 64, 3, padding=1)
+        self.bn_decode2 = nn.BatchNorm2d(64)
+        self.upconv3 = nn.ConvTranspose2d(64, 48, 2, 2)
+        self.bn_upconv3 = nn.BatchNorm2d(48)
+        self.conv_decode3 = nn.Conv2d(48, 48, 3, padding=1)
+        self.bn_decode3 = nn.BatchNorm2d(48)
+        if half_sized_output:
+          self.upconv4 = nn.Identity()
+          self.bn_upconv4 = nn.Identity()
+          self.conv_decode4 = nn.Conv2d(48, output_channels, 3, padding=1)
         else:
-            if small_decoder:
-                self.upconv1 = nn.ConvTranspose2d(512,128,2,2)
-                self.bn_upconv1 = nn.BatchNorm2d(128)
-                self.conv_decode1 = nn.Conv2d(128, 128, 3,padding=1)
-                self.bn_decode1 = nn.BatchNorm2d(128)
-                self.upconv2 = nn.ConvTranspose2d(128,64,2,2)
-                self.bn_upconv2 = nn.BatchNorm2d(64)
-                self.conv_decode2 = nn.Conv2d(64, 64, 3,padding=1)
-                self.bn_decode2 = nn.BatchNorm2d(64)
-                self.upconv3 = nn.ConvTranspose2d(64,48,2,2)
-                self.bn_upconv3 = nn.BatchNorm2d(48)
-                self.conv_decode3 = nn.Conv2d(48, 48, 3,padding=1)
-                self.bn_decode3 = nn.BatchNorm2d(48)
-                if half_sized_output:
-                    self.upconv4 = nn.Identity()
-                    self.bn_upconv4 = nn.Identity()
-                    self.conv_decode4 = nn.Conv2d(48, output_channels, 3,padding=1)
-                else:
-                    self.upconv4 = nn.ConvTranspose2d(48,32,2,2)
-                    self.bn_upconv4 = nn.BatchNorm2d(32)
-                    self.conv_decode4 = nn.Conv2d(32, output_channels, 3,padding=1)
-            else:
-                self.upconv1 = nn.ConvTranspose2d(512,256,2,2)
-                self.bn_upconv1 = nn.BatchNorm2d(256)
-                self.conv_decode1 = nn.Conv2d(256, 256, 3,padding=1)
-                self.bn_decode1 = nn.BatchNorm2d(256)
-                self.upconv2 = nn.ConvTranspose2d(256,128,2,2)
-                self.bn_upconv2 = nn.BatchNorm2d(128)
-                self.conv_decode2 = nn.Conv2d(128, 128, 3,padding=1)
-                self.bn_decode2 = nn.BatchNorm2d(128)
-                self.upconv3 = nn.ConvTranspose2d(128,96,2,2)
-                self.bn_upconv3 = nn.BatchNorm2d(96)
-                self.conv_decode3 = nn.Conv2d(96, 96, 3,padding=1)
-                self.bn_decode3 = nn.BatchNorm2d(96)
-                if half_sized_output:
-                    self.upconv4 = nn.Identity()
-                    self.bn_upconv4 = nn.Identity()
-                    self.conv_decode4 = nn.Conv2d(96, output_channels, 3,padding=1)
-                else:
-                    self.upconv4 = nn.ConvTranspose2d(96,64,2,2)
-                    self.bn_upconv4 = nn.BatchNorm2d(64)
-                    self.conv_decode4 = nn.Conv2d(64, output_channels, 3,padding=1)
-
-
-
-
-    def forward(self,representation):
-        if self.num_classes is None:
-            x = self.upconv1(representation)
-            x = self.bn_upconv1(x)
-            x = self.relu(x)
-            x = self.conv_decode1(x)
-            x = self.bn_decode1(x)
-            x = self.relu(x)
-            x = self.upconv2(x)
-            x = self.bn_upconv2(x)
-            x = self.relu(x)
-            x = self.conv_decode2(x)
-
-            x = self.bn_decode2(x)
-            x = self.relu(x)
-            x = self.upconv3(x)
-            x = self.bn_upconv3(x)
-            x = self.relu(x)
-            x = self.conv_decode3(x)
-            x = self.bn_decode3(x)
-            x = self.relu(x)
-            if not self.half_sized_output:
-                x = self.upconv4(x)
-                x = self.bn_upconv4(x)
-                x = self.relu(x)
-            x = self.conv_decode4(x)
-
+          self.upconv4 = nn.ConvTranspose2d(48, 32, 2, 2)
+          self.bn_upconv4 = nn.BatchNorm2d(32)
+          self.conv_decode4 = nn.Conv2d(32, output_channels, 3, padding=1)
+      else:
+        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, 2)
+        self.bn_upconv1 = nn.BatchNorm2d(256)
+        self.conv_decode1 = nn.Conv2d(256, 256, 3, padding=1)
+        self.bn_decode1 = nn.BatchNorm2d(256)
+        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
+        self.bn_upconv2 = nn.BatchNorm2d(128)
+        self.conv_decode2 = nn.Conv2d(128, 128, 3, padding=1)
+        self.bn_decode2 = nn.BatchNorm2d(128)
+        self.upconv3 = nn.ConvTranspose2d(128, 96, 2, 2)
+        self.bn_upconv3 = nn.BatchNorm2d(96)
+        self.conv_decode3 = nn.Conv2d(96, 96, 3, padding=1)
+        self.bn_decode3 = nn.BatchNorm2d(96)
+        if half_sized_output:
+          self.upconv4 = nn.Identity()
+          self.bn_upconv4 = nn.Identity()
+          self.conv_decode4 = nn.Conv2d(96, output_channels, 3, padding=1)
         else:
-            x = self.block12(representation)
-
-            x = self.conv3(x)
-            x = self.bn3(x)
-            x = self.relu(x)
-
-            x = self.conv4(x)
-            x = self.bn4(x)
-            x = self.relu(x)
-
-            x = F.adaptive_avg_pool2d(x, (1, 1))
-            x = x.view(x.size(0), -1)
-            x = self.fc(x)
-        return x
+          self.upconv4 = nn.ConvTranspose2d(96, 64, 2, 2)
+          self.bn_upconv4 = nn.BatchNorm2d(64)
+          self.conv_decode4 = nn.Conv2d(64, output_channels, 3, padding=1)
+
+  def forward(self, representation):
+    if self.num_classes is None:
+      x = self.upconv1(representation)
+      x = self.bn_upconv1(x)
+      x = self.relu(x)
+      x = self.conv_decode1(x)
+      x = self.bn_decode1(x)
+      x = self.relu(x)
+      x = self.upconv2(x)
+      x = self.bn_upconv2(x)
+      x = self.relu(x)
+      x = self.conv_decode2(x)
+
+      x = self.bn_decode2(x)
+      x = self.relu(x)
+      x = self.upconv3(x)
+      x = self.bn_upconv3(x)
+      x = self.relu(x)
+      x = self.conv_decode3(x)
+      x = self.bn_decode3(x)
+      x = self.relu(x)
+      if not self.half_sized_output:
+        x = self.upconv4(x)
+        x = self.bn_upconv4(x)
+        x = self.relu(x)
+      x = self.conv_decode4(x)
 
+    else:
+      x = self.block12(representation)
 
+      x = self.conv3(x)
+      x = self.bn3(x)
+      x = self.relu(x)
+
+      x = self.conv4(x)
+      x = self.bn4(x)
+      x = self.relu(x)
+
+      x = F.adaptive_avg_pool2d(x, (1, 1))
+      x = x.view(x.size(0), -1)
+      x = self.fc(x)
+    return x
 
 
 class XceptionTaskonomy(nn.Module):
-    """
+  """
     Xception optimized for the ImageNet dataset, as specified in
     https://arxiv.org/pdf/1610.02357.pdf
     """
-    def __init__(self,size=1, tasks=None,num_classes=None, ozan=False,half_sized_output=False):
-        """ Constructor
-        Args:
-            num_classes: number of classes
-        """
-        super(XceptionTaskonomy, self).__init__()
-        pre_rep_size=728
-        sizes=[32,64,128,256,728,728,728,728,728,728,728,728,728]
-        if size == 1:
-            sizes=[32,64,128,256,728,728,728,728,728,728,728,728,728]
-        elif size==.2:
-            sizes=[16,32,64,256,320,320,320,320,320,320,320,320,320]
-        elif size==.3:
-            sizes=[32,64,128,256,728,728,728,728,728,728,728,728,728]
-        elif size==.4:
-            sizes=[32,64,128,256,728,728,728,728,728,728,728,728,728]
-        elif size==.5:
-            sizes=[24,48,96,192,512,512,512,512,512,512,512,512,512]
-        elif size==.8:
-            sizes=[32,64,128,248,648,648,648,648,648,648,648,648,648]
-        elif size==2:
-            sizes=[32,64, 128,256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
-        elif size==4:
-            sizes=[64,128,256,512,1456,1456,1456,1456,1456,1456,1456,1456,1456]
-
-
-        self.encoder=Encoder(sizes=sizes)
-        pre_rep_size=sizes[-1]
-
-        self.tasks=tasks
-        self.ozan=ozan
-        self.task_to_decoder = {}
-
-
-
-        if tasks is not None:
-
-            self.final_conv = SeparableConv2d(pre_rep_size,512,3,1,1)
-            self.final_conv_bn = nn.BatchNorm2d(512)
-            for task in tasks:
-                if task == 'segment_semantic':
-                    output_channels = 18
-                if task == 'depth_zbuffer':
-                    output_channels = 1
-                if task == 'normal':
-                    output_channels = 3
-                if task == 'edge_occlusion':
-                    output_channels = 1
-                if task == 'keypoints2d':
-                    output_channels = 1
-                if task == 'edge_texture':
-                    output_channels = 1
-                if task == 'reshading':
-                    output_channels = 1
-                if task == 'rgb':
-                    output_channels = 3
-                if task == 'principal_curvature':
-                    output_channels = 2
-                decoder=Decoder(output_channels,half_sized_output=half_sized_output)
-                self.task_to_decoder[task]=decoder
-        else:
-            self.task_to_decoder['classification']=Decoder(output_channels=0,num_classes=1000)
 
-        self.decoders = nn.ModuleList(self.task_to_decoder.values())
+  def __init__(self,
+               size=1,
+               tasks=None,
+               num_classes=None,
+               ozan=False,
+               half_sized_output=False):
+    """ Constructor
 
-        #------- init weights --------
-        for m in self.modules():
-            if isinstance(m, nn.Conv2d):
-                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
-                m.weight.data.normal_(0, math.sqrt(2. / n))
-            elif isinstance(m, nn.BatchNorm2d):
-                m.weight.data.fill_(1)
-                m.bias.data.zero_()
-        #-----------------------------
-
-
-    def forward(self, input):
-        rep = self.encoder(input)
-
-
-        if self.tasks is None:
-            return self.decoders[0](rep)
-
-        rep = self.final_conv(rep)
-        rep = self.final_conv_bn(rep)
-
-        outputs={'rep':rep}
-        if self.ozan:
-            OzanRepFunction.n=len(self.decoders)
-            rep = ozan_rep_function(rep)
-            for i,(task,decoder) in enumerate(zip(self.task_to_decoder.keys(),self.decoders)):
-                outputs[task]=decoder(rep[i])
-        else:
-            TrevorRepFunction.n=len(self.decoders)
-            rep = trevor_rep_function(rep)
-            for i,(task,decoder) in enumerate(zip(self.task_to_decoder.keys(),self.decoders)):
-                outputs[task]=decoder(rep)
-
-        return outputs
+        Args:
+            num_classes: number of classes
+    """
+    super(XceptionTaskonomy, self).__init__()
+    pre_rep_size = 728
+    sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
+    if size == 1:
+      sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
+    elif size == .2:
+      sizes = [16, 32, 64, 256, 320, 320, 320, 320, 320, 320, 320, 320, 320]
+    elif size == .3:
+      sizes = [16, 32, 96, 128, 312, 312, 312, 312, 312, 312, 312, 312, 312]
+    elif size == .4:
+      sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
+    elif size == .5:
+      sizes = [24, 48, 96, 192, 512, 512, 512, 512, 512, 512, 512, 512, 512]
+    elif size == .8:
+      sizes = [32, 64, 128, 248, 648, 648, 648, 648, 648, 648, 648, 648, 648]
+    elif size == 2:
+      sizes = [32, 64, 128, 256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
+    elif size == 3:
+      sizes = [48, 96, 196, 384, 1092, 1092, 1092, 1092, 1092, 1092, 1092, 1092, 1092]
+    elif size == 4:
+      sizes = [
+          64, 128, 256, 512, 1456, 1456, 1456, 1456, 1456, 1456, 1456, 1456,
+          1456
+      ]
+
+    self.encoder = Encoder(sizes=sizes)
+    pre_rep_size = sizes[-1]
+
+    self.tasks = tasks
+    self.ozan = ozan
+    self.task_to_decoder = {}
+
+    if tasks is not None:
+
+      self.final_conv = SeparableConv2d(pre_rep_size, 512, 3, 1, 1)
+      self.final_conv_bn = nn.BatchNorm2d(512)
+      for task in tasks:
+        if task == 'segment_semantic':
+          output_channels = 18
+        if task == 'depth_zbuffer':
+          output_channels = 1
+        if task == 'normal':
+          output_channels = 3
+        if task == 'edge_occlusion':
+          output_channels = 1
+        if task == 'keypoints2d':
+          output_channels = 1
+        if task == 'edge_texture':
+          output_channels = 1
+        if task == 'reshading':
+          output_channels = 1
+        if task == 'rgb':
+          output_channels = 3
+        if task == 'principal_curvature':
+          output_channels = 2
+        decoder = Decoder(output_channels, half_sized_output=half_sized_output)
+        self.task_to_decoder[task] = decoder
+    else:
+      self.task_to_decoder['classification'] = Decoder(
+          output_channels=0, num_classes=1000)
+
+    self.decoders = nn.ModuleList(self.task_to_decoder.values())
+
+    #------- init weights --------
+    for m in self.modules():
+      if isinstance(m, nn.Conv2d):
+        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
+        m.weight.data.normal_(0, math.sqrt(2. / n))
+      elif isinstance(m, nn.BatchNorm2d):
+        m.weight.data.fill_(1)
+        m.bias.data.zero_()
+    #-----------------------------
+
+  count = 0
+
+  def input_per_task_losses(self, losses):
+    # if GradNormRepFunction.inital_task_losses is None:
+    #     GradNormRepFunction.inital_task_losses=losses
+    #     GradNormRepFunction.current_weights=[1 for i in losses]
+    XceptionTaskonomy.count += 1
+    if XceptionTaskonomy.count < 200:
+      GradNormRepFunction.inital_task_losses = losses
+      GradNormRepFunction.current_weights = [1 for i in losses]
+    elif XceptionTaskonomy.count % 20 == 0:
+      with open('gradnorm_weights.txt', 'a') as myfile:
+        myfile.write(
+            str(XceptionTaskonomy.count) + ': ' +
+            str(GradNormRepFunction.current_weights) + '\n')
+      # logging.info(str(GradNormRepFunction.current_weights))
+      # logging.info(str(XceptionTaskonomy.count))
+    GradNormRepFunction.current_task_losses = losses
+
+  def forward(self, input):
+    rep = self.encoder(input)
+
+    if self.tasks is None:
+      return self.decoders[0](rep)
+
+    rep = self.final_conv(rep)
+    rep = self.final_conv_bn(rep)
+
+    outputs = {'rep': rep}
+    if self.ozan == 'gradnorm':
+      GradNormRepFunction.n = len(self.decoders)
+      rep = gradnorm_rep_function(rep)
+      for i, (task, decoder) in enumerate(
+          zip(self.task_to_decoder.keys(), self.decoders)):
+        outputs[task] = decoder(rep[i])
+    elif self.ozan:
+      OzanRepFunction.n = len(self.decoders)
+      rep = ozan_rep_function(rep)
+      for i, (task, decoder) in enumerate(
+          zip(self.task_to_decoder.keys(), self.decoders)):
+        outputs[task] = decoder(rep[i])
+    else:
+      TrevorRepFunction.n = len(self.decoders)
+      rep = trevor_rep_function(rep)
+      for i, (task, decoder) in enumerate(
+          zip(self.task_to_decoder.keys(), self.decoders)):
+        outputs[task] = decoder(rep)
 
+    return outputs
 
 
 def xception_taskonomy_new(**kwargs):
-    """
+  """
     Construct Xception.
     """
 
-    model = XceptionTaskonomy(**kwargs,size=1)
+  model = XceptionTaskonomy(**kwargs, size=3)
+
+  return model
 
-    return model
 
 def xception_taskonomy_new_fifth(**kwargs):
-    """
+  """
     Construct Xception.
     """
 
-    model = XceptionTaskonomy(**kwargs,size=.2)
+  model = XceptionTaskonomy(**kwargs, size=.2)
+
+  return model
 
-    return model
 
 def xception_taskonomy_new_quad(**kwargs):
-    """
+  """
     Construct Xception.
     """
 
-    model = XceptionTaskonomy(**kwargs,size=4)
+  model = XceptionTaskonomy(**kwargs, size=4)
+
+  return model
 
-    return model
 
 def xception_taskonomy_new_half(**kwargs):
-    """
+  """
     Construct Xception.
     """
 
-    model = XceptionTaskonomy(**kwargs,size=.5)
+  model = XceptionTaskonomy(**kwargs, size=.5)
+
+  return model
 
-    return model
 
 def xception_taskonomy_new_80(**kwargs):
-    """
+  """
     Construct Xception.
     """
 
-    model = XceptionTaskonomy(**kwargs,size=.8)
+  model = XceptionTaskonomy(**kwargs, size=.8)
+
+  return model
 
-    return model
 
 def xception_taskonomy_ozan(**kwargs):
-    """
+  """
     Construct Xception.
     """
 
-    model = XceptionTaskonomy(ozan=True,**kwargs)
+  model = XceptionTaskonomy(ozan=True, **kwargs)
 
-    return model
+  return model
+
+def xception_taskonomy_new_gradnorm(ozan='gradnorm', **kwargs):
+  """
+    Construct Xception.
+    """
+  model = XceptionTaskonomySmall(ozan='gradnorm', **kwargs)
+  return model