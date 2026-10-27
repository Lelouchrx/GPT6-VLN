
import cv2
import numpy as np

class RGBPose:
    def __init__(self):
        self.commanded_yaw=0
        self.visual_yaw=0.
        self.unknown_translations=0
        self.last=None

    def update(self,before,after,action):
        self.commanded_yaw=(self.commanded_yaw+(15 if action==2 else -15 if action==3 else 0)+180)%360-180
        result={'rotation_matrix':None,'translation_direction_camera':None,
                'translation_scale':'unknown','translation_m':None,'quality':'unknown',
                'source':'monocular RGB feature tracking + essential matrix RANSAC'}
        if action==0:return result
        a=cv2.cvtColor(np.asarray(before['rgb'])[...,:3],cv2.COLOR_RGB2GRAY)
        b=cv2.cvtColor(np.asarray(after['rgb'])[...,:3],cv2.COLOR_RGB2GRAY)
        p=cv2.goodFeaturesToTrack(a,800,.01,8)
        if p is not None and len(p)>=30:
            q,valid,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(31,31),maxLevel=4)
            if q is not None:
                back,ok,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(31,31),maxLevel=4)
                if back is not None:
                    keep=valid.ravel().astype(bool)&ok.ravel().astype(bool)&(np.linalg.norm(back-p,axis=2).ravel()<1.)
                    p0=p.reshape(-1,2)[keep];p1=q.reshape(-1,2)[keep]
                    if len(p0)>=30:
                        flow=float(np.median(np.linalg.norm(p1-p0,axis=1)))
                        result.update(tracked_features=len(p0),median_pixel_flow=round(flow,3))
                        h,w=a.shape;f=w/2
                        K=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]],dtype=float)

                        if action==1 and flow>=.7:
                            E,mask=cv2.findEssentialMat(p0,p1,K,method=cv2.RANSAC,prob=.999,threshold=1.)
                            if E is not None and E.shape==(3,3):
                                n,R,t,inliers=cv2.recoverPose(E,p0,p1,K,mask=mask)
                                ratio=n/len(p0)
                                if n>=25 and ratio>=.45:
                                    displacement=(-R.T@t).ravel()
                                    yaw=float(np.degrees(np.arctan2(R[0,2],R[2,2])))
                                    self.visual_yaw+=yaw
                                    result.update(rotation_matrix=R.round(5).tolist(),
                                        translation_direction_camera=displacement.round(5).tolist(),
                                        inliers=int(n),inlier_ratio=round(ratio,3),
                                        visual_relative_yaw_deg=round(yaw,3),quality='relative_direction_only')
                        elif action==1 and flow<.7:
                            result['quality']='low_parallax_translation_unknown'
        if action==1:self.unknown_translations+=1
        self.last=result
        return result

    def context(self):
        return {'source':'RGB-only relative pose estimates, no simulator pose or depth',
            'commanded_heading_deg':self.commanded_yaw,
            'visual_yaw_accumulated_deg':round(self.visual_yaw,3),
            'last_relative_pose':self.last,
            'metric_position_m':None,'metric_scale':'unknown',
            'unknown_translation_scale_intervals':self.unknown_translations,
            'limitations':'Monocular scale, pure rotation, texture and parallax ambiguity. Commanded turn is not measured yaw. No metric trajectory inferred from nominal step length.'}
