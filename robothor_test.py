from ai2thor.controller import Controller

from PIL import Image

controller = Controller(
    agentMode="locobot",
    visibilityDistance=1.5,
    scene="FloorPlan16_physics",
    gridSize=0.25,
    movementGaussianSigma=0.005,
    rotateStepDegrees=90,
    rotateGaussianSigma=0.5,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    width=300,
    height=300,
    fieldOfView=60
)

user = input("Waiting for input ....")
event = controller.last_event

while(user != "end"):

    if(user[0] == "r"):
        event = controller.step(
            action="RotateRight",
            degrees=int(user[1:])
        )
    if(user[0] == "l"):
        event = controller.step(
            action="RotateLeft",
            degrees=int(user[1:])
        )
    if(user[0] == "f"):
        event = controller.step(
            action="MoveAhead",
            moveMagnitude=0.25
        )
    if(user[0] == "b"):
        event = controller.step(
            action ="MoveBack",
            moveMagnitude=0.25
        )
    if(user[0] == "i"):
        img_pil = Image.fromarray(event.frame)
        img_pil2 = Image.fromarray(event.cv2img) 
 
        img_pil.save("output_image_frame.png")
        img_pil2.save("output_image_cv2.png")
        
    user = input("Enter command: ")

