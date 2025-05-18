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
zoom_offset = 0
objects = []
offset_x = 0.
face_count = 0
vertice_3_face_count = 0
vertice_4_face_count = 0
other = 0
draw_count = 0

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;  // interpolated normal

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 light_pos;

void main()
{
    // light and material properties
    
    vec3 light_color = vec3(1,1,1);
    vec3 material_color = vec3(1.0, 1.0, 1.0);
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = vec3(1,1,1);  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    FragColor = vec4(color, 1.);
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
    global g_ctrl_pressed, g_shift_pressed, g_alt_pressed, g_dragging, zoom_offset
    global g_cam_ang, g_cam_height, g_cam_elevation, g_cam_offset_x, g_cam_offset_y
    global g_prev_x, g_prev_y
    
    dx = xpos - g_prev_x
    dy = ypos - g_prev_y
    if g_dragging:
        if g_shift_pressed and g_alt_pressed:
            if dx > 0:
                g_cam_offset_x += .05
            elif dx < 0:
                g_cam_offset_x -= .05
            elif dy > 0:
                g_cam_offset_y += .05
            elif dy < 0:
                g_cam_offset_y -= .05
        elif g_ctrl_pressed and g_alt_pressed:
            if dy > 0:
                zoom_offset += .05
            elif dy < 0:
                zoom_offset -= .05
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
    global face_count, vertice_3_face_count, vertice_4_face_count, other, draw_count
    vertices = []
    normals = []
    indices = []
    normal_indices = []
    vertex_array = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == "v":
                vertex = [float(x) for x in parts[1:]]
                vertices.append(vertex)
            
            elif parts[0] == "vn":
                normal = [float(x) for x in parts[1:]]
                normals.append(normal)
            
            elif parts[0] == 'f':
                face_count += 1
                if len(parts) > 5:
                    other += 1
                    verts = [p.split('//') for p in parts[1:]]
                    for i in range(1, len(verts) - 1):
                        for j in [0, i, i+1]:
                            v_index, n_index = map(int, verts[j])
                            vertex = vertices[v_index - 1]
                            normal = normals[n_index - 1]
                            vertex_array.append(vertex + normal)
                            indices.append(len(vertex_array) - 1)

                elif len(parts) == 5:
                    vertice_4_face_count += 1
                    verts = [p.split('//') for p in parts[1:]]
                    for j in [0, 1, 2]:
                        v_index, n_index = map(int, verts[j])
                        vertex = vertices[v_index - 1]
                        normal = normals[n_index - 1]
                        vertex_array.append(vertex + normal)
                        indices.append(len(vertex_array) - 1)
                    
                    for j in [0, 2, 3]:
                        v_index, n_index = map(int, verts[j])
                        vertex = vertices[v_index - 1]
                        normal = normals[n_index - 1]
                        vertex_array.append(vertex + normal)
                        indices.append(len(vertex_array) - 1)

                elif len(parts) == 4:
                    vertice_3_face_count += 1
                    for p in parts[1:]:
                        v_index, n_index = map(int, p.split('//'))
                        vertex = vertices[v_index - 1]
                        normal = normals[n_index - 1]
                        vertex_array.append(vertex + normal)
                        indices.append(len(vertex_array) - 1)


    vertex_array = np.array(vertex_array, dtype=np.float32)
    draw_count = len(vertex_array)
    indices = np.array(indices, dtype=np.uint32)

    file_name = [file_path.split('\\')[-1]]
    print(f"Obj file name : {file_name}, Total number of faces : {face_count}, Number of faces with 3 vertices : {vertice_3_face_count}, Number of faces with 4 vertices : {vertice_4_face_count}, Number of faces with more than 4 vertices : {other}")
    
    return vertex_array, indices

def setup_buffers(vertices_array, indices):
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    # VBO 설정
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices_array.nbytes, vertices_array, GL_STATIC_DRAW)

    # 정점 속성 설정
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_TRUE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

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
        vertices_array, indices = load_obj(model_path)
        VAO, index_count = setup_buffers(vertices_array, indices)
        
        position = (offset_x, 0., 0.)
        objects.append({
            "VAO": VAO,
            "index_count": index_count,
            "position": position
        })
        
        offset_x += 2.0
    except Exception as e:
        print(f"Failed to load model: {e}")

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position       #normal        # color
         -5.0, 0.0, 0.0, 1.0, 0.0, 0.0, # x-axis start
         5.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, -5.0, 0.0, 0.0, 1.0, 0.0, # y-axis start
         0.0, 5.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, -5.0, 0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 5.0,  0.0, 0.0, 1.0, # z-axis end 
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
    step = 0.5
    range_val = 20.0

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
    loc_view_pos = glGetUniformLocation(shader_program, 'view_pos')
    loc_M = glGetUniformLocation(shader_program, 'M')
    loc_light_pos = glGetUniformLocation(shader_program, "light_pos")
    
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
        P_ortho = glm.ortho(-1,1,-1,1,-1,1)
        P = glm.perspective(45, 1, .1 , 20)
        
        # view matrix
        # rotate camera position with g_cam_ang / move camera up & down with g_cam_height
        view_pos = glm.vec3(g_cam_offset_x + .5*np.sin(g_cam_ang),g_cam_offset_y + .5*np.sin(g_cam_elevation),.5*np.cos(g_cam_ang))
        light_pos = glm.vec3(0, 10, 0)
        view_target = glm.vec3(g_cam_offset_x ,g_cam_offset_y ,0)
        direction = (view_pos - view_target) / glm.length(view_pos - view_target)
        view_pos += zoom_offset * direction
        V = glm.lookAt(view_pos, view_target, glm.vec3(0,1,0))
        #view_pos = glm.vec3(5*np.sin(g_cam_ang),.1,5*np.cos(g_cam_ang))
        #V = glm.lookAt(view_pos, glm.vec3(0,0,0), glm.vec3(0,1,0))
        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniform3f(loc_view_pos, view_pos.x, view_pos.y, view_pos.z)
        glUniform3f(loc_light_pos, light_pos.x, light_pos.y, light_pos.z)
        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)
        MVP_grid = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_grid))
        glUniformMatrix4fv(loc_M, 1, GL_FALSE, glm.value_ptr(I))
        glBindVertexArray(vao_grid)
        glDrawArrays(GL_LINES, 0, grid_vertex_count)
        # swap front and back buffers
        for obj in objects:
            model = glm.mat4(1.0)
            #model = glm.scale(model, glm.vec3(0.1))
            model = glm.translate(model, glm.vec3(obj["position"]))
            #model = glm.rotate(model, glfwGetTime(), glm.vec3(0, 1, 0))
            model = P*V*model
            glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(model))

            glBindVertexArray(obj["VAO"])
            glDrawArrays(GL_TRIANGLES, 0, draw_count)
            glBindVertexArray(0)
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()