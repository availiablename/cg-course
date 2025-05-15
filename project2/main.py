from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_cam_ang = 0.
g_cam_elevation = 0.
g_cam_offset_x =  0.
g_cam_offset_y =  0.
g_shift_pressed = False
g_alt_pressed = False
g_ctrl_pressed = False
g_dragging = False
g_prev_x = 0
g_prev_y = 0
g_zoom = 1.0
objects = []
offset_x = 0.

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program


def key_callback(window, key, scancode, action, mods):
    global g_cam_ang, g_cam_height
    global g_shift_pressed, g_alt_pressed, g_ctrl_pressed
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if key==GLFW_KEY_LEFT_SHIFT:
        if action == GLFW_PRESS:
            g_shift_pressed = True
        elif action == GLFW_RELEASE:
            g_shift_pressed = False
    if key==GLFW_KEY_LEFT_ALT:
        if action == GLFW_PRESS:
            g_alt_pressed = True
        elif action == GLFW_RELEASE:
            g_alt_pressed = False
    if key==GLFW_KEY_LEFT_CONTROL:
        if action == GLFW_PRESS:
            g_ctrl_pressed = True
        elif action == GLFW_RELEASE:
            g_ctrl_pressed = False

def mouse_button_callback(window, button, action, mods):
    global g_dragging, g_cursor_pos
    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action == GLFW_PRESS or action==GLFW_REPEAT:
            g_dragging = True
        elif action == GLFW_RELEASE:
            g_dragging = False

def cursor_position_callback(window, xpos, ypos):
    global g_ctrl_pressed, g_shift_pressed, g_alt_pressed, g_zoom, g_dragging
    global g_cam_ang, g_cam_height, g_cam_elevation, g_cam_offset_x, g_cam_offset_y
    global g_prev_x, g_prev_y
    dx = xpos - g_prev_x
    dy = ypos - g_prev_y
    
    if g_dragging:
        if g_shift_pressed and g_alt_pressed:
            if dx > 0:
                g_cam_offset_x += .01
            elif dx < 0:
                g_cam_offset_x -= .01
            elif dy > 0:
                g_cam_offset_y += .01
            elif dy < 0:
                g_cam_offset_y -= .01
        elif g_ctrl_pressed and g_alt_pressed:
            if dy < 0:
                g_zoom /= 1.05
            elif dy > 0:
                g_zoom *= 1.05
        elif g_alt_pressed:
            if dx > 0:
                g_cam_ang += np.radians(1)
            elif dx < 0:
                g_cam_ang += np.radians(-1)
            elif dy > 0:
                g_cam_elevation += np.radians(1)
            elif dy < 0:
                g_cam_elevation -= np.radians(-1)

    g_prev_x = xpos
    g_prev_y = ypos

def load_obj(file_path):
    vertices = []
    normals = []
    indices = []
    normal_indices = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # 정점 (v)
            if parts[0] == "v":
                vertex = [float(x) for x in parts[1:]]
                vertices.extend(vertex)
            
            # 법선 벡터 (vn)
            elif parts[0] == "vn":
                normal = [float(x) for x in parts[1:]]
                normals.extend(normal)
            
            # 면 정의 (f)
            elif parts[0] == "f":
                for p in parts[1:]:
                    # v/vt/vn or v//vn or v
                    indices_split = p.split('/')
                    
                    # 정점 인덱스
                    vertex_index = int(indices_split[0]) - 1
                    
                    indices.append(vertex_index)
                    
                    # 법선 벡터 인덱스 (v//vn or v/vt/vn)
                    if len(indices_split) == 3 or (len(indices_split) == 2 and indices_split[1] != ''):
                        normal_index = int(indices_split[-1]) - 1
                        normal_indices.append(normal_index)

    # numpy 배열로 변환
    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    normal_indices = np.array(normal_indices, dtype=np.uint32)

    print(f"Loaded {len(vertices) // 3} vertices, {len(normals) // 3} normals, {len(indices)} indices")
    return vertices, indices

def setup_buffers(vertices, indices):
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    # VBO 설정
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # EBO 설정
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # 정점 속성 설정
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_TRUE, 3 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return VAO, len(indices)

def drop_callback(window, paths):
    global objects
    global offset_x
    model_path = paths[0]
    print(f"Loading model: {model_path}")
    for path in paths:
        print(f"Dropped file: {path}")
    try:
        vertices, indices = load_obj(model_path)
        VAO, index_count = setup_buffers(vertices, indices)
        
        # 랜덤 위치로 모델 배치
        position = (offset_x, 0., 0.)
        objects.append({
            "VAO": VAO,
            "index_count": index_count,
            "position": position
        })
        print(f"Successfully loaded {model_path} with {index_count} indices")
        offset_x += 2.0
    except Exception as e:
        print(f"Failed to load model: {e}")

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         1.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, 1.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, 0.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 1.0,  0.0, 0.0, 1.0, # z-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid():
    grid_vertices = []
    step = 0.1
    range_val = 3.0

    for x in np.arange(-range_val, range_val + step, step):
        grid_vertices.extend([x, 0.0, -range_val, 0.5, 0.5, 0.5])  
        grid_vertices.extend([x, 0.0, range_val, 0.5, 0.5, 0.5])   

    for z in np.arange(-range_val, range_val + step, step):
        grid_vertices.extend([-range_val, 0.0, z, 0.5, 0.5, 0.5])
        grid_vertices.extend([range_val, 0.0, z, 0.5, 0.5, 0.5])

    grid_vertices = np.array(grid_vertices, dtype=np.float32)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, grid_vertices.nbytes, grid_vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * grid_vertices.itemsize))
    glEnableVertexAttribArray(1)

    return VAO, len(grid_vertices)

def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(1200, 1200, '2021085923', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetDropCallback(window, drop_callback);

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_grid, grid_vertex_count = prepare_vao_grid()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # projection matrix
        # use orthogonal projection (we'll see details later)
        #P = glm.ortho(-1*g_zoom,1*g_zoom,-1*g_zoom,1*g_zoom,-1,1)
        P = glm.perspective(45, 1, .1, 20)
        
        # view matrix
        # rotate camera position with g_cam_ang / move camera up & down with g_cam_height
        V = glm.lookAt(glm.vec3(g_cam_offset_x + .5*np.sin(g_cam_ang),g_cam_offset_y + .5*np.sin(g_cam_elevation),.5*np.cos(g_cam_ang)), glm.vec3(g_cam_offset_x ,g_cam_offset_y ,0), glm.vec3(0,1,0))
        #view_pos = glm.vec3(5*np.sin(g_cam_ang),.1,5*np.cos(g_cam_ang))
        #V = glm.lookAt(view_pos, glm.vec3(0,0,0), glm.vec3(0,1,0))
        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)
        MVP_grid = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_grid))
        glBindVertexArray(vao_grid)
        glDrawArrays(GL_LINES, 0, grid_vertex_count)
        # swap front and back buffers
        for obj in objects:
            model = glm.mat4(1.0)
            model = glm.scale(model, glm.vec3(0.1))
            model = glm.translate(model, glm.vec3(obj["position"]))
            #model = glm.rotate(model, glfwGetTime(), glm.vec3(0, 1, 0))
            model = P*V*model
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(model))

            glBindVertexArray(obj["VAO"])
            glDrawElements(GL_TRIANGLES, obj["index_count"], GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()