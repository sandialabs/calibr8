cmake_minimum_required(VERSION 3.15...3.21.1)

set(CAPP_TRUE TRUE)
#this regex is designed to match a Git ref so that the remote
#is the first sub-expression and the branch is the second
set(CAPP_REMOTE_REF_REGEX "refs/remotes/([0-9A-Za-z_-]+)/([0-9A-Za-z\\./_-]+)")

if (WIN32)
  set(CMAKE_PROGRAM_PATH "C:/Program Files") #workaround a workaround for MSVC 2017 in FindGit.cmake
endif()
find_package(Git REQUIRED QUIET)
find_package(Python COMPONENTS Interpreter QUIET)

function(capp_stdout msg)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "${msg}")
endfunction()

function(capp_list_to_string)
  cmake_parse_arguments(PARSE_ARGV 0 capp_list_to_string "" "LIST;STRING" "")
  set(str)
  foreach(item IN LISTS "${capp_list_to_string_LIST}")
    if (str)
      set(str "${str} ${item}")
    else()
      set(str "${item}")
    endif()
  endforeach()
  set(${capp_list_to_string_STRING} "${str}" PARENT_SCOPE)
endfunction()

function(capp_get_subdirectories result curdir)
  file(GLOB children RELATIVE "${curdir}" "${curdir}/*")
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY "${curdir}/${child}")
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist} PARENT_SCOPE)
endfunction()

function(capp_subdirectories)
  cmake_parse_arguments(PARSE_ARGV 0 capp_subdirectories "" "PARENT_DIRECTORY;RESULT_VARIABLE" "")
  file(GLOB children RELATIVE "${capp_subdirectories_PARENT_DIRECTORY}" "${capp_subdirectories_PARENT_DIRECTORY}/*")
  set(dirlist "")
  foreach (child ${children})
    if (IS_DIRECTORY "${capp_subdirectories_PARENT_DIRECTORY}/${child}")
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${capp_subdirectories_RESULT_VARIABLE} ${dirlist} PARENT_SCOPE)
endfunction()

function(capp_execute)
  cmake_parse_arguments(PARSE_ARGV
    0
    capp_execute
    "OUTPUT_QUIET;ERROR_QUIET;OUTPUT_STRIP_TRAILING_WHITESPACE"
    "WORKING_DIRECTORY;RESULT_VARIABLE;OUTPUT_VARIABLE;ERROR_VARIABLE"
    "COMMAND")
  set(extra_args)
  if (capp_execute_OUTPUT_VARIABLE)
    list(APPEND extra_args OUTPUT_VARIABLE capp_execute_output)
  endif()
  if (capp_execute_ERROR_VARIABLE)
    list(APPEND extra_args ERROR_VARIABLE capp_execute_error)
  endif()
  if (capp_execute_OUTPUT_QUIET)
    list(APPEND extra_args OUTPUT_QUIET)
  endif()
  if (capp_execute_ERROR_QUIET)
    list(APPEND extra_args ERROR_QUIET)
  endif()
  if (capp_execute_OUTPUT_STRIP_TRAILING_WHITESPACE)
    list(APPEND extra_args OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  execute_process(
      COMMAND ${capp_execute_COMMAND}
      WORKING_DIRECTORY "${capp_execute_WORKING_DIRECTORY}"
      RESULT_VARIABLE capp_execute_result
      ${extra_args}
  )
  if (capp_execute_OUTPUT_VARIABLE)
    set(${capp_execute_OUTPUT_VARIABLE} "${capp_execute_output}" PARENT_SCOPE)
  endif()
  if (capp_execute_ERROR_VARIABLE)
    set(${capp_execute_ERROR_VARIABLE} "${capp_execute_error}" PARENT_SCOPE)
  endif()
  set(${capp_execute_RESULT_VARIABLE} "${capp_execute_result}" PARENT_SCOPE)
endfunction()

function(capp_add_paths result var list)
  string(REPLACE ":" ";" contents "$ENV{${var}}")
  foreach(path IN LISTS list)
    if (NOT "${path}" IN_LIST contents)
      list(PREPEND contents "${path}")
    endif()
  endforeach()
  string(REPLACE ";" ":" contents "${contents}")
  set(${result} "${contents}" PARENT_SCOPE)
endfunction()

function(capp_remove_paths result var list)
  string(REPLACE ":" ";" contents "$ENV{${var}}")
  foreach(path IN LISTS list)
    list(REMOVE_ITEM contents "${path}")
  endforeach()
  string(REPLACE ";" ":" contents "${contents}")
  set(${result} "${contents}" PARENT_SCOPE)
endfunction()

function(capp_add_file)
  cmake_parse_arguments(PARSE_ARGV 0 capp_add_file "" "RESULT_VARIABLE;FILE" "")
  capp_execute(
    COMMAND "${GIT_EXECUTABLE}" add "${capp_add_file_FILE}"
    WORKING_DIRECTORY "${CAPP_ROOT}"
    RESULT_VARIABLE git_add_result
  )
  set(${capp_add_file_RESULT_VARIABLE} ${git_add_result} PARENT_SCOPE)
endfunction()

function(capp_commit)
  cmake_parse_arguments(PARSE_ARGV 0 capp_commit "" "RESULT_VARIABLE;MESSAGE" "")
  capp_execute(
    COMMAND "${GIT_EXECUTABLE}" commit -m "${capp_commit_MESSAGE}"
    WORKING_DIRECTORY "${CAPP_ROOT}"
    RESULT_VARIABLE git_commit_result
  )
  set(${capp_commit_RESULT_VARIABLE} ${git_commit_result} PARENT_SCOPE)
endfunction()

function(capp_changes_committed)
  cmake_parse_arguments(PARSE_ARGV 0 capp_changes_committed "" "PACKAGE;RESULT_VARIABLE;OUTPUT_VARIABLE" "")
  set(package ${capp_changes_committed_PACKAGE})
  set(result_variable ${capp_changes_committed_RESULT_VARIABLE})
  set(output_variable ${capp_changes_committed_OUTPUT_VARIABLE})
  if (${package}_IGNORE_UNCOMMITTED)
    set(${result_variable} 0 PARENT_SCOPE)
    return()
  endif()
  capp_execute(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain
    WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
    RESULT_VARIABLE git_uncommitted_result
    OUTPUT_VARIABLE git_uncommitted_output)
  if (NOT git_uncommitted_result EQUAL 0)
    set(${result_variable} ${git_uncommitted_result} PARENT_SCOPE)
    return()
  endif()
  if (git_uncommitted_output)
    set(${output_variable} "${git_uncommitted_output}" PARENT_SCOPE)
    set(${result_variable} -1 PARENT_SCOPE)
    return()
  endif()
  set(${result_variable} 0 PARENT_SCOPE)
endfunction()

#This function's main purpose is to get the repository at source/<package>
#to be at the commit specified in package/<package>/package.cmake
#(technically, whatever is already in ${package}_COMMIT and ${package}_GIT_URL)
#It also aims to do this as nicely as possible.
#For example, if that commit is the head of a branch, then the repository
#should be at that branch and tracking the appropriate remote branch.

function(capp_checkout)
  cmake_parse_arguments(PARSE_ARGV 0 capp_checkout "" "PACKAGE;RESULT_VARIABLE" "")
  set(package ${capp_checkout_PACKAGE})
  set(desired_commit ${${package}_COMMIT})
  set(desired_git_url ${${package}_GIT_URL})
  set(has_submodules ${${package}_HAS_SUBMODULES})
  #Safety check: make sure there are not uncommitted changes, otherwise the
  #package can't really be checked out to the desired commit.
  capp_changes_committed(
    PACKAGE ${package}
    RESULT_VARIABLE uncommitted_result
    OUTPUT_VARIABLE uncommitted_output)
  if (NOT uncommitted_result EQUAL 0)
    message("\nCApp refusing to check out ${package} because it has uncommitted changes:\n${uncommitted_output}")
    set(${capp_checkout_RESULT_VARIABLE} -1 PARENT_SCOPE)
    return()
  endif()
  #The common case is that the repository is already
  #at this commit. In that case, just exit early.
  capp_get_package_commit(
    PACKAGE ${package}
    COMMIT_VARIABLE current_commit
    RESULT_VARIABLE get_commit_result
    )
  if (NOT get_commit_result EQUAL 0)
    message("CApp: failed to get the current commit of ${package}")
    set(${capp_checkout_RESULT_VARIABLE} ${get_commit_result} PARENT_SCOPE)
    return()
  endif()
  if (current_commit STREQUAL desired_commit)
    message("CApp: ${package} is already at the desired commit")
  else()
    #From here on out we will be trying to change the commit that a package
    #is checked out to, so we will mark it as needing re-configuration
    capp_invalidate_config(${package})
    #For multiple reasons (either fetching or picking a branch to check out),
    #we need to identify which remote the desired URL maps to.
    set(remote)
    #In order to do that, we begin by gathering a list of existing remotes.
    capp_execute(
      COMMAND "${GIT_EXECUTABLE}" remote show
      WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
      RESULT_VARIABLE remote_show_result
      OUTPUT_VARIABLE remote_show_output)
    if (NOT remote_show_result EQUAL 0)
      message("git remote show failed on ${package}")
      set(${capp_checkout_RESULT_VARIABLE} ${remote_show_result} PARENT_SCOPE)
      return()
    endif()
    #Replace any newlines with a semicolon,
    #thus converting a one-line-per-remote string into a CMake list
    string(REGEX REPLACE "[\r\n]+" ";" existing_remotes "${remote_show_output}")
    list(REMOVE_ITEM existing_remotes "")
    message("CApp: in ${package}, existing_remotes=${existing_remotes}")
    #Then we check if any of these remotes match the desired URL:
    foreach(existing_remote IN LISTS existing_remotes)
      capp_get_remote_url(
        PACKAGE ${package}
        REMOTE ${existing_remote}
        GIT_URL_VARIABLE existing_remote_url
        RESULT_VARIABLE remote_url_result)
      if (NOT remote_show_result EQUAL 0)
        message("failed to get URL for remote ${remote} of ${package}")
        set(${capp_checkout_RESULT_VARIABLE} ${remote_url_result} PARENT_SCOPE)
        return()
      endif()
      message("CApp: ${package} remote ${existing_remote} has URL ${existing_remote_url}")
      if (existing_remote_url STREQUAL desired_git_url)
        message("CApp: ${package} remote ${existing_remote} matches desired URL ${desired_git_url}")
        set(remote ${existing_remote})
      endif()
    endforeach()
    if (NOT remote)
      #If we are here, then none of the existing remotes match the desired Git URL.
      #In that case, CApp will go so far as to try to add it for you with a reasonable name.
      #That reasonable name will be essentially the GitHub/GitLab organization/group path.
      #The following crazy regex is just trying to extract the first component of the
      #remote repository path.
      string(REGEX REPLACE
        ".*(:|\\.[a-z]+/)([A-Za-z0-9_-]+)/[A-Za-z0-9/_-]+(\\.git)?"
        "\\2"
        reasonable_remote_name
        "${desired_git_url}")
      message("CApp: a reasonable name for the remote for ${desired_git_url} is ${reasonable_remote_name}")
      if (reasonable_remote_name IN_LIST existing_remotes)
        #If this reasonable name is already one of the remotes, let's just give up.
        #It is best for the user to decide how to resolve this mess.
        message("\nCApp wanted to add a remote ${reasonable_remote_name} with URL ${desired_git_url} to ${package} but it already exists.\n")
        set(${capp_checkout_RESULT_VARIABLE} -1 PARENT_SCOPE)
        return()
      endif()
      #We have a reasonable name and it isn't one of the remotes yet, let's add it
      capp_execute(
        COMMAND "${GIT_EXECUTABLE}" remote add ${reasonable_remote_name} ${desired_git_url}
        WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
        RESULT_VARIABLE remote_add_result)
      if (NOT remote_add_result EQUAL 0)
        message("failed to add remote ${reasonable_remote_name} with URL ${desired_git_url} to ${package}")
        set(${capp_checkout_RESULT_VARIABLE} ${remote_add_result} PARENT_SCOPE)
        return()
      endif()
      message("CApp: added remote ${reasonable_remote_name} with URL ${desired_git_url} to ${package}")
      set(remote ${reasonable_remote_name})
    endif()
    #If we are here, then the repository is not at the desired commit.
    #Performance optimization: check to see if the commit exists in the
    #local repository first before going to the fetching step, because
    #fetching may not be possible in some cases (in which case we still
    #want this operation to succeed if it can) and is expensive.
    capp_execute(
      COMMAND "${GIT_EXECUTABLE}" cat-file -e ${desired_commit}^{commit}
      WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
      OUTPUT_QUIET
      ERROR_QUIET
      RESULT_VARIABLE commit_exists_result)
    if (NOT commit_exists_result EQUAL 0)
      #If we are here, then the desired commit doesn't exist locally.
      #In that case, the next step
      #is to ensure the desired Git URL exists as a remote and has been fetched.
      message("CApp: ${package} desired commit ${desired_commit} doesn't exist locally")
      #If we are here, then ${remote} is a remote with the right Git URL.
      #Let's fetch it.
      set(shallow_failed FALSE)
      message("CApp: shallow fetching refs of package ${package}")
      capp_execute(
        COMMAND "${GIT_EXECUTABLE}" fetch --depth 1 ${remote}
        WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
        OUTPUT_QUIET
        ERROR_QUIET
        RESULT_VARIABLE refs_fetch_result)
      if (NOT refs_fetch_result EQUAL 0)
        message("CApp: failed to shallow fetch refs of ${package}")
        set(shallow_failed TRUE)
      endif()
      message("CApp: shallow fetching commit ${desired_commit} of ${package}")
      capp_execute(
        COMMAND "${GIT_EXECUTABLE}" fetch --depth 1 ${remote}
                ${desired_commit}
        WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
        OUTPUT_QUIET
        ERROR_QUIET
        RESULT_VARIABLE commit_fetch_result)
      if (NOT commit_fetch_result EQUAL 0)
        message("CApp: failed to shallow fetch commit ${desired_commit} of ${package}")
        set(shallow_failed TRUE)
      endif()
      if (shallow_failed)
        message("CApp: failed to shallow fetch ${package}")
        message("CApp: fully fetching remote ${remote} of ${package}")
        capp_execute(
          COMMAND "${GIT_EXECUTABLE}" fetch ${remote}
          WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
          OUTPUT_QUIET
          ERROR_QUIET
          RESULT_VARIABLE fetch_result)
        if (NOT fetch_result EQUAL 0)
          message("CApp: failed to fetch remote ${remote} of ${package}")
          set(${capp_checkout_RESULT_VARIABLE} ${fetch_result} PARENT_SCOPE)
          return()
        endif()
      endif()
      message("CApp: succeeded in fetching remote ${remote} of ${package}")
      #Then let's check if the given commit exists now.
      capp_execute(
        COMMAND "${GIT_EXECUTABLE}" cat-file -e ${desired_commit}^{commit}
        OUTPUT_QUIET
        ERROR_QUIET
        WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
        RESULT_VARIABLE commit_exists_result)
      if (NOT commit_exists_result EQUAL 0)
        #If it doesn't, then this is user error and the desired commit isn't in the
        #desired Git URL.
        message("\nCApp can't find commit ${desired_commit} of ${package} even after fetching from ${desired_git_url}\n")
        set(${capp_checkout_RESULT_VARIABLE} -1 PARENT_SCOPE)
        return()
      endif()
    endif()
    #If we're here, then the desired commit does exist in the local Git repository
    #for the package.
    #Now, we begin the process of checking it out as nicely as possible.
    #The main theme of this niceness is to check out a branch if possible.
    #So, the first step is to see if any remote branches (actualy refs) on the desired
    #remote point to the desired commit.
    capp_execute(
      COMMAND "${GIT_EXECUTABLE}" for-each-ref --points-at=${desired_commit}
      WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
      OUTPUT_VARIABLE pointing_refs_output
      RESULT_VARIABLE pointing_refs_result)
    if (NOT pointing_refs_result EQUAL 0)
      message("git for-each-ref --points-at=${desired_commit} failed for ${package}")
      set(${capp_checkout_RESULT_VARIABLE} ${pointing_refs_result} PARENT_SCOPE)
      return()
    endif()
    #Use MATCHALL to extract a CMake list of all the remote refs that point to the
    #desired commit.
    string(REGEX MATCHALL "refs/remotes/${remote}/[^ \t\r\n]+" pointing_refs "${pointing_refs_output}")
    #One of these refs can be HEAD, which is special and not a branch name
    list(REMOVE_ITEM pointing_refs "refs/remotes/${remote}/HEAD")
    message("CApp: in ${package}, pointing_refs=${pointing_refs}")
    if (pointing_refs)
      #At this point, there are some remote refs that point to the desired commit.
      #Let's try to pick one of those as the branch to check out.
      #Honestly, I can't think of much more than preferring the typical default
      #branch names and after that just choosing the first one on the list.
      if ("refs/remotes/${remote}/master" IN_LIST pointing_refs)
        set(branch "master")
      elseif ("refs/remotes/${remote}/main" IN_LIST pointing_refs)
        set(branch "main")
      else()
        list(GET pointing_refs 0 pointing_ref)
        string(REGEX REPLACE "${CAPP_REMOTE_REF_REGEX}" "\\2" branch "${pointing_ref}")
      endif()
      message("CApp: in ${package}, picked branch ${branch}")
      #Okay, we finally have a remote and a branch that we want to check out.
      #Let's do it!
      capp_execute(
        COMMAND "${GIT_EXECUTABLE}" checkout -B ${branch} --track ${remote}/${branch}
        WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
        OUTPUT_QUIET
        ERROR_QUIET
        RESULT_VARIABLE branch_checkout_result)
      if (NOT branch_checkout_result EQUAL 0)
        message("git checkout -B ${branch} --track ${remote}/${branch} failed in ${package}")
        set(${capp_checkout_RESULT_VARIABLE} ${branch_checkout_result} PARENT_SCOPE)
        return()
      endif()
      message("CApp: in ${package}, checked out ${branch}, tracking ${remote}/${branch}")
    else()
      #If there are no remote refs that point to this commit, then it is a
      #"detached HEAD" situation and we can just check out the desired commit.
      capp_execute(
        COMMAND "${GIT_EXECUTABLE}" checkout ${desired_commit}
        WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
        OUTPUT_QUIET
        ERROR_QUIET
        RESULT_VARIABLE detached_checkout_result)
      if (NOT detached_checkout_result EQUAL 0)
        message("git checkout ${desired_commit} failed in ${package}")
        set(${capp_checkout_RESULT_VARIABLE} ${detached_checkout_result} PARENT_SCOPE)
        return()
      endif()
      message("CApp: checked out commit ${desired_commit} explicitly")
    endif()
  endif()
  if(has_submodules)
    #Aaaand by now we've succeeded in "git checkout"'ing a good thing.
    #But we're not done yet! Submodules!
    capp_execute(
      COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive
      WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
      RESULT_VARIABLE submodule_result
      )
    if (NOT submodule_result EQUAL 0)
      message("git submodule update --init --recursive failed")
      set(${capp_checkout_RESULT_VARIABLE} "${submodule_result}" PARENT_SCOPE)
      return()
    endif()
    message("CApp: submodule update of ${package} completed")
  endif()
  message("CApp: checkout of ${package} succeeded")
  #We did it. We "checked out a commit".
  set(${capp_checkout_RESULT_VARIABLE} 0 PARENT_SCOPE)
endfunction()

function(capp_clone)
  cmake_parse_arguments(PARSE_ARGV 0 capp_clone "" "PACKAGE;RESULT_VARIABLE" "")
  file(MAKE_DIRECTORY "${CAPP_SOURCE_ROOT}")
  set(cmd_list
      "${GIT_EXECUTABLE}"
      clone --depth 1
      "${${capp_clone_PACKAGE}_GIT_URL}"
      ${capp_clone_PACKAGE})
  message("\nCApp: shallow cloning ${capp_clone_PACKAGE} from ${${capp_clone_PACKAGE}_GIT_URL}\n")
  capp_execute(
    COMMAND ${cmd_list}
    WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}"
    RESULT_VARIABLE git_clone_result
    OUTPUT_QUIET
    )
  if (NOT git_clone_result EQUAL 0)
    capp_list_to_string(LIST cmd_list STRING cmd_string)
    message("\nCApp: shallow clone of ${capp_clone_PACKAGE} failed.\nCommand was: ${cmd_string}\n")
    file(REMOVE_RECURSE "${CAPP_SOURCE_ROOT}/${capp_clone_PACKAGE}")
    set(${capp_clone_RESULT_VARIABLE} "${git_clone_result}" PARENT_SCOPE)
    return()
  endif()
  capp_checkout(
    PACKAGE ${capp_clone_PACKAGE}
    RESULT_VARIABLE checkout_result)
  if (NOT checkout_result EQUAL 0)
    message("\nCApp: checkout of ${capp_clone_PACKAGE} failed\n")
    set(${capp_clone_RESULT_VARIABLE} "${checkout_result}" PARENT_SCOPE)
    return()
  endif()
  message("CApp: clone of ${capp_clone_PACKAGE} succeeded\n")
  set(${capp_clone_RESULT_VARIABLE} 0 PARENT_SCOPE)
  set(${capp_clone_PACKAGE}_IS_CLONED TRUE PARENT_SCOPE)
endfunction()

function(capp_ensure_venv)
  if (NOT EXISTS "${CAPP_VENV_ROOT}")
    message("CApp: Python virtual environment at ${CAPP_VENV_ROOT} is needed but doesn't exist")
    if (NOT Python_FOUND)
      message(FATAL_ERROR "CApp: Cannot create Python virtual environment because Python was not found")
    endif()
    message("CApp: Creating new Python virtual environment at ${CAPP_VENV_ROOT} using ${Python_EXECUTABLE}")
    set(cmd_list 
        "${Python_EXECUTABLE}"
        -m
        venv
        "${CAPP_VENV_ROOT}"
       )
    capp_execute(
        COMMAND ${cmd_list}
        RESULT_VARIABLE venv_result
        )
    if (NOT venv_result EQUAL 0)
      capp_list_to_string(LIST cmd_list STRING cmd_string)
      file(REMOVE_RECURSE "${CAPP_VENV_ROOT}")
      message(FATAL_ERROR "CApp: Failed to create Python virtual environment.\ncommand: ${cmd_string}\n")
    else()
      message("CApp: Successfully created Python virtual environment at ${CAPP_VENV_ROOT}")
    endif()
    set(cmd_list 
        "${CAPP_VENV_ROOT}/bin/pip"
        install
        ${CAPP_PIP_FLAGS}
        --upgrade
        wheel
        setuptools
        pip
       )
    capp_list_to_string(LIST cmd_list STRING cmd_string)
    capp_execute(
        COMMAND ${cmd_list}
        RESULT_VARIABLE pip_result
        )
    if (NOT ${pip_result} EQUAL 0)
      file(REMOVE_RECURSE "${CAPP_VENV_ROOT}")
      message(FATAL_ERROR "CApp: Failed to upgrade wheel, setuptools, and pip in Python virtual environment.\nThis can be due to proxy issues, try passing the --proxy flag to CApp.\ncommand: ${cmd_string}\n")
      return()
    else()
      message("CApp: Successfully upgraded wheel, setuptools, and pip in Python virtual environment")
    endif()
  endif()
endfunction()

function(capp_configure)
  cmake_parse_arguments(PARSE_ARGV 0 capp_configure "" "PACKAGE;RESULT_VARIABLE" "")
  if (${capp_configure_PACKAGE}_PYTHON_DEPENDENCIES)
    capp_ensure_venv()
    capp_list_to_string(LIST ${capp_configure_PACKAGE}_PYTHON_DEPENDENCIES STRING depstring)
    message("CApp: Installing Python dependencies ${depstring} of ${capp_configure_PACKAGE} using pip")
    set(cmd_list
        "${CAPP_VENV_ROOT}/bin/pip"
        install
        ${CAPP_PIP_FLAGS}
        ${${capp_configure_PACKAGE}_PYTHON_DEPENDENCIES}
       )
    capp_list_to_string(LIST cmd_list STRING cmd_string)
    capp_execute(
        COMMAND ${cmd_list}
        RESULT_VARIABLE pip_result
        )
    if (NOT ${pip_result} EQUAL 0)
      set(${capp_configure_RESULT_VARIABLE} ${pip_result} PARENT_SCOPE)
      message("CApp: failed to install Python dependencies of ${capp_configure_PACKAGE}\ncommand: ${cmd_string}\n")
      return()
    else()
      message("CApp: successfully installed Python dependencies ${depstring} of ${capp_configure_PACKAGE}") 
    endif()
  endif()
  file(MAKE_DIRECTORY "${CAPP_BUILD_ROOT}/${capp_configure_PACKAGE}")
  set(source_directory "${CAPP_SOURCE_ROOT}/${capp_configure_PACKAGE}")
  if (${capp_configure_PACKAGE}_SUBDIRECTORY)
    set(source_directory "${source_directory}/${${capp_configure_PACKAGE}_SUBDIRECTORY}")
  endif()
  set(cmakelists_path "${source_directory}/CMakeLists.txt")
  set(setup_path "${source_directory}/setup.py")
  set(pyproject_path "${source_directory}/pyproject.toml")
  if (EXISTS "${cmakelists_path}")
    set(options "-DCMAKE_INSTALL_PREFIX=${CAPP_INSTALL_ROOT}/${capp_configure_PACKAGE}")
    if (NOT WIN32)
      list(APPEND options "-DCMAKE_BUILD_TYPE=${${capp_configure_PACKAGE}_BUILD_TYPE}")
    endif()
    list(APPEND options ${${capp_configure_PACKAGE}_OPTIONS})
    capp_list_to_string(LIST options STRING print_options)
    message("\nCApp configuring ${capp_configure_PACKAGE} with these options: ${print_options}\n")
    capp_execute(
        COMMAND
        "${CMAKE_COMMAND}"
        "${source_directory}"
        ${options}
        WORKING_DIRECTORY "${CAPP_BUILD_ROOT}/${capp_configure_PACKAGE}"
        RESULT_VARIABLE cmake_configure_result
    )
    set(${capp_configure_RESULT_VARIABLE} "${cmake_configure_result}" PARENT_SCOPE)
    if (cmake_configure_result EQUAL 0)
      set(${capp_configure_PACKAGE}_IS_CONFIGURED TRUE PARENT_SCOPE)
      file(WRITE "${CAPP_BUILD_ROOT}/${capp_configure_PACKAGE}/capp_configured.txt" "Yes")
    endif()
  elseif(EXISTS "${setup_path}" OR EXISTS "${pyproject_path}")
    set(${capp_configure_RESULT_VARIABLE} 0 PARENT_SCOPE)
    set(${capp_configure_PACKAGE}_IS_CONFIGURED TRUE PARENT_SCOPE)
    file(WRITE "${CAPP_BUILD_ROOT}/${capp_configure_PACKAGE}/capp_configured.txt" "Yes")
  else()
    set(${capp_configure_RESULT_VARIABLE} -1 PARENT_SCOPE)
    message("\nCApp: none of the following exist:\n${cmakelists_path}\n${setup_path}\n${pyproject_path}\n")
  endif()
endfunction()

function(capp_build)
  cmake_parse_arguments(PARSE_ARGV 0 capp_build "" "PACKAGE;RESULT_VARIABLE" "ARGUMENTS")
  set(with_args "")
  if (capp_build_ARGUMENTS)
    capp_list_to_string(LIST capp_build_ARGUMENTS STRING print_args)
    set(with_args " with extra arguments ${print_args}")
  endif()
  message("\nCApp building ${capp_build_PACKAGE}${with_args}\n")
  set(source_directory "${CAPP_SOURCE_ROOT}/${capp_build_PACKAGE}")
  if (${capp_build_PACKAGE}_SUBDIRECTORY)
    set(source_directory "${source_directory}/${${capp_build_PACKAGE}_SUBDIRECTORY}")
  endif()
  set(cmakelists_path "${source_directory}/CMakeLists.txt")
  if (EXISTS "${cmakelists_path}")
    capp_execute(
        COMMAND
        "${CMAKE_COMMAND}"
        "--build"
        "."
        "--config"
        ${${capp_build_PACKAGE}_BUILD_TYPE}
        ${capp_build_ARGUMENTS}
        WORKING_DIRECTORY "${CAPP_BUILD_ROOT}/${capp_build_PACKAGE}"
        RESULT_VARIABLE cmake_build_result
    )
    set(${capp_build_RESULT_VARIABLE} "${cmake_build_result}" PARENT_SCOPE)
  else()
    set(${capp_build_RESULT_VARIABLE} 0 PARENT_SCOPE)
  endif()
endfunction()

function(capp_install)
  cmake_parse_arguments(PARSE_ARGV 0 capp_install "" "PACKAGE;RESULT_VARIABLE" "")
  message("\nCApp installing ${capp_install_PACKAGE}\n")
  set(source_directory "${CAPP_SOURCE_ROOT}/${capp_install_PACKAGE}")
  if (${capp_install_PACKAGE}_SUBDIRECTORY)
    set(source_directory "${source_directory}/${${capp_install_PACKAGE}_SUBDIRECTORY}")
  endif()
  set(cmakelists_path "${source_directory}/CMakeLists.txt")
  set(setup_path "${source_directory}/setup.py")
  set(pyproject_path "${source_directory}/pyproject.toml")
  if (EXISTS "${cmakelists_path}")
    capp_execute(
        COMMAND
        "${CMAKE_COMMAND}"
        "--install"
        "."
        "--config"
        ${${capp_install_PACKAGE}_BUILD_TYPE}
        WORKING_DIRECTORY "${CAPP_BUILD_ROOT}/${capp_install_PACKAGE}"
        RESULT_VARIABLE cmake_install_result
    )
    set(${capp_install_RESULT_VARIABLE} "${cmake_install_result}" PARENT_SCOPE)
  elseif(EXISTS "${setup_path}" OR EXISTS "${pyproject_path}")
    capp_ensure_venv()
    message("CApp: Installing ${capp_install_PACKAGE} using pip")
    set(cmd_list
        "${CAPP_VENV_ROOT}/bin/pip"
        install
        ${CAPP_PIP_FLAGS}
        "${source_directory}"
       )
    capp_list_to_string(LIST cmd_list STRING cmd_string)
    capp_execute(
        COMMAND ${cmd_list}
        RESULT_VARIABLE pip_result
        )
    set(${capp_install_RESULT_VARIABLE} ${pip_result} PARENT_SCOPE)
    if (NOT pip_result EQUAL 0)
      message("CApp: failed to install ${capp_install_PACKAGE} with pip\ncommand: ${cmd_string}\n")
    endif()
  else()
    set(${capp_install_RESULT_VARIABLE} -1 PARENT_SCOPE)
    message("CApp: none of the following exist:\n${cmakelists_path}\n${setup_path}")
  endif()
endfunction()

macro(capp_app)
  cmake_parse_arguments(capp_app "" "BUILD_TYPE" "ROOT_PACKAGES" ${ARGN})
  set(CAPP_ROOT_PACKAGES "${capp_app_ROOT_PACKAGES}")
  set(build_type "${capp_app_BUILD_TYPE}")
  if (NOT build_type)
    set(build_type Release)
  endif()
  if (NOT CAPP_BUILD_TYPE)
    set(CAPP_BUILD_TYPE "${build_type}")
  endif()
endmacro()

function(capp_package)
  cmake_parse_arguments(PARSE_ARGV 0 arg
      "NO_CONFIGURE_CACHE;IGNORE_UNCOMMITTED;HAS_SUBMODULES;IS_LOCAL"
      "GIT_URL;COMMIT;SUBDIRECTORY;BUILD_TYPE"
      "OPTIONS;DEPENDENCIES;PYTHON_DEPENDENCIES;PYTHONPATH")
  set(${CAPP_PACKAGE}_GIT_URL ${arg_GIT_URL} PARENT_SCOPE)
  set(${CAPP_PACKAGE}_COMMIT ${arg_COMMIT} PARENT_SCOPE)
  set(${CAPP_PACKAGE}_OPTIONS "${arg_OPTIONS}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_DEPENDENCIES "${arg_DEPENDENCIES}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_PYTHON_DEPENDENCIES "${arg_PYTHON_DEPENDENCIES}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_PYTHONPATH "${arg_PYTHONPATH}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_NO_CONFIGURE_CACHE "${arg_NO_CONFIGURE_CACHE}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_IGNORE_UNCOMMITTED "${arg_IGNORE_UNCOMMITTED}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_HAS_SUBMODULES "${arg_HAS_SUBMODULES}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_IS_LOCAL "${arg_IS_LOCAL}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_SUBDIRECTORY "${arg_SUBDIRECTORY}" PARENT_SCOPE)
  if (arg_BUILD_TYPE)
    set(${CAPP_PACKAGE}_BUILD_TYPE "${arg_BUILD_TYPE}" PARENT_SCOPE)
  else()
    set(${CAPP_PACKAGE}_BUILD_TYPE "${CAPP_BUILD_TYPE}" PARENT_SCOPE)
  endif()
endfunction()

function(capp_topsort_packages)
  set(unsorted_list "${CAPP_PACKAGES}")
  set(sorted_list)
  set(no_incoming_set)
  foreach (package IN LISTS unsorted_list)
    set(${package}_dependers)
  endforeach()
  foreach (package IN LISTS unsorted_list)
    set(${package}_dependees "${${package}_DEPENDENCIES}")
    foreach (dependee IN LISTS ${package}_dependees)
      list(APPEND ${dependee}_dependers ${package})
    endforeach()
    if (NOT ${package}_dependees)
      list(APPEND no_incoming_set ${package})
    endif()
  endforeach()
  while (no_incoming_set)
    list(POP_FRONT no_incoming_set package_n)
    list(APPEND sorted_list ${package_n})
    while (${package_n}_dependers)
      list(POP_FRONT ${package_n}_dependers depender)
      list(REMOVE_ITEM ${depender}_dependees ${package_n})
      if (NOT ${depender}_dependees)
        list(APPEND no_incoming_set ${depender})
        list(REMOVE_DUPLICATES no_incoming_set)
      endif()
    endwhile()
  endwhile()
  set(bad_edges)
  foreach (package IN LISTS unsorted_list)
    foreach (dependee IN LISTS ${package}_dependees)
      set(bad_edges "${bad_edges}${dependee} -> ${package}\n")
    endforeach()
  endforeach()
  if (bad_edges)
    message(FATAL_ERROR "There is a cycle in the dependency graph involving:\n${bad_edges}")
  endif()
  set(CAPP_PACKAGES "${sorted_list}" PARENT_SCOPE)
endfunction()

function(capp_build_install)
  cmake_parse_arguments(PARSE_ARGV 0 capp_build_install "" "PACKAGE;RESULT_VARIABLE" "BUILD_ARGUMENTS")
  capp_build(
    PACKAGE ${capp_build_install_PACKAGE}
    ARGUMENTS
    ${capp_build_install_BUILD_ARGUMENTS}
    RESULT_VARIABLE capp_build_result
  )
  if (NOT capp_build_result EQUAL 0)
    set(${capp_build_install_RESULT_VARIABLE} ${capp_build_result} PARENT_SCOPE)
    message("CApp: capp_build_install failed because capp_build failed")
    return()
  endif()
  capp_install(
    PACKAGE ${capp_build_install_PACKAGE}
    RESULT_VARIABLE capp_install_result
  )
  set(${capp_build_install_RESULT_VARIABLE} ${capp_install_result} PARENT_SCOPE)
  if (capp_install_result EQUAL 0)
    set(${capp_build_install_PACKAGE}_IS_INSTALLED TRUE PARENT_SCOPE)
    file(WRITE "${CAPP_BUILD_ROOT}/${capp_build_install_PACKAGE}/capp_installed.txt" "Yes")
  else()
    message("CApp: capp_build_install failed because capp_install failed")
  endif()
endfunction()

function(capp_read_package_file)
  cmake_parse_arguments(PARSE_ARGV 0 capp_read_package_file "" "PACKAGE" "")
  set(CAPP_PACKAGE ${capp_read_package_file_PACKAGE})
  set(capp_read_package_file_path "${CAPP_PACKAGE_ROOT}/${CAPP_PACKAGE}/package.cmake")
  include("${capp_read_package_file_path}")
  set(${CAPP_PACKAGE}_NO_CONFIGURE_CACHE ${${CAPP_PACKAGE}_NO_CONFIGURE_CACHE} PARENT_SCOPE)
  set(${CAPP_PACKAGE}_IGNORE_UNCOMMITTED ${${CAPP_PACKAGE}_IGNORE_UNCOMMITTED} PARENT_SCOPE)
  set(${CAPP_PACKAGE}_HAS_SUBMODULES ${${CAPP_PACKAGE}_HAS_SUBMODULES} PARENT_SCOPE)
  set(${CAPP_PACKAGE}_IS_LOCAL ${${CAPP_PACKAGE}_IS_LOCAL} PARENT_SCOPE)
  set(${CAPP_PACKAGE}_GIT_URL ${${CAPP_PACKAGE}_GIT_URL} PARENT_SCOPE)
  set(${CAPP_PACKAGE}_COMMIT ${${CAPP_PACKAGE}_COMMIT} PARENT_SCOPE)
  set(${CAPP_PACKAGE}_OPTIONS "${${CAPP_PACKAGE}_OPTIONS}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_DEPENDENCIES "${${CAPP_PACKAGE}_DEPENDENCIES}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_PYTHON_DEPENDENCIES "${${CAPP_PACKAGE}_PYTHON_DEPENDENCIES}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_PYTHONPATH "${${CAPP_PACKAGE}_PYTHONPATH}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_SUBDIRECTORY "${${CAPP_PACKAGE}_SUBDIRECTORY}" PARENT_SCOPE)
  set(${CAPP_PACKAGE}_BUILD_TYPE "${${CAPP_PACKAGE}_BUILD_TYPE}" PARENT_SCOPE)
  set(CAPP_PACKAGES ${CAPP_PACKAGES} ${CAPP_PACKAGE} PARENT_SCOPE)
endfunction()

macro(capp_find_root)
  while (CAPP_TRUE)
    get_filename_component(CAPP_ROOT_PARENT "${CAPP_ROOT}" DIRECTORY)
    if (CAPP_ROOT_PARENT STREQUAL CAPP_ROOT)
      message(FATAL_ERROR "CApp could not find app.cmake in ${CMAKE_CURRENT_SOURCE_DIR} or any parent directories: Run capp init first")
    endif()
    if (EXISTS "${CAPP_ROOT}/app.cmake")
      break()
    endif()
    set(CAPP_ROOT "${CAPP_ROOT_PARENT}")
  endwhile()
  set(CAPP_SOURCE_ROOT "${CAPP_ROOT}/source")
  set(CAPP_PACKAGE_ROOT "${CAPP_ROOT}/package")
  capp_get_commit(
      GIT_REPO_PATH "${CAPP_ROOT}"
      COMMIT_VARIABLE CAPP_COMMIT
      RESULT_VARIABLE get_commit_result)
  if (NOT get_commit_result EQUAL 0)
    message(FATAL_ERROR "CApp could not query the commit of the build repository")
  endif()
endmacro()

macro(capp_setup_flavor)
  cmake_parse_arguments(arg "OPTIONAL" "" "" ${ARGN})
  if (NOT CAPP_FLAVOR)
    set(flavor_dir_regex "${CAPP_ROOT}/flavor/([^/])")
    if (CMAKE_CURRENT_SOURCE_DIR MATCHES "${flavor_dir_regex}")
      string(REGEX REPLACE "${flavor_dir_regex}" "\\1" flavor "${CMAKE_CURRENT_SOURCE_DIR}")
      set(CAPP_FLAVOR "${flavor}")
    endif()
  endif()
  if (NOT CAPP_FLAVOR)
    if (DEFINED ENV{CAPP_FLAVOR})
      set(CAPP_FLAVOR $ENV{CAPP_FLAVOR})
    endif()
  endif()
  if (CAPP_FLAVOR)
    set(CAPP_FLAVOR_ROOT "${CAPP_ROOT}/flavor/${CAPP_FLAVOR}")
    set(CAPP_BUILD_ROOT "${CAPP_FLAVOR_ROOT}/build")
    if (NOT CAPP_PREFIX)
      if (DEFINED ENV{CAPP_PREFIX})
        set(CAPP_PREFIX $ENV{CAPP_PREFIX})
      endif()
    endif()
    if (CAPP_PREFIX)
      set(CAPP_INSTALL_ROOT "${CAPP_PREFIX}")
      set(CAPP_VENV_ROOT "${CAPP_PREFIX}/venv")
    else()
      set(CAPP_INSTALL_ROOT "${CAPP_FLAVOR_ROOT}/install")
      set(CAPP_VENV_ROOT "${CAPP_FLAVOR_ROOT}/venv")
    endif()
    set(flavor_file "${CAPP_FLAVOR_ROOT}/flavor.cmake")
    if (NOT EXISTS "${flavor_file}")
      message(FATAL_ERROR "CApp: Flavor file ${flavor_file} doesn't exist. It should define any CMake variables needed to setup this flavor of the overall build. If you have no such variables, creating an empty file will suffice.")
    endif()
    include("${flavor_file}")
  else()
    if (NOT arg_OPTIONAL)
      capp_subdirectories(PARENT_DIRECTORY "${CAPP_ROOT}/flavor" RESULT_VARIABLE flavors)
      string(REPLACE ";" ", " flavors "${flavors}")
      message(FATAL_ERROR "CApp: No flavor has been selected. You can select a flavor by setting the CAPP_FLAVOR environment variable, giving the `-f <flavor>` command line flag to CApp, or by running CApp from inside a flavor subdirectory. Available flavors are: ${flavors}.")
    endif()
  endif()
endmacro()

macro(capp_recursive_read_package_file package)
  list(FIND CAPP_PACKAGES ${package} list_index)
  if (list_index EQUAL -1)
    capp_read_package_file(PACKAGE ${package})
    foreach(dependency IN LISTS ${package}_DEPENDENCIES)
      capp_recursive_read_package_file(${dependency})
    endforeach()
  endif()
endmacro()

macro(capp_read_package_files_by_dependency)
  include("${CAPP_ROOT}/app.cmake")
  set(CAPP_PACKAGES)
  foreach(root_package IN LISTS CAPP_ROOT_PACKAGES)
    capp_recursive_read_package_file(${root_package})
  endforeach()
endmacro()

macro(capp_read_all_package_files)
  include("${CAPP_ROOT}/app.cmake")
  set(CAPP_PACKAGES)
  file(
    GLOB packages
    LIST_DIRECTORIES true
    RELATIVE "${CAPP_PACKAGE_ROOT}"
    "${CAPP_PACKAGE_ROOT}/*")
  foreach(package IN LISTS packages)
    capp_read_package_file(PACKAGE ${package})
  endforeach()
endmacro()

function(capp_invalidate_config package)
  if (${package}_NO_CONFIGURE_CACHE)
    file(REMOVE "${CAPP_BUILD_ROOT}/${package}/CMakeCache.txt")
  endif()
  file(REMOVE "${CAPP_BUILD_ROOT}/${package}/capp_configured.txt")
endfunction()

function(capp_invalidate_install package)
  file(REMOVE "${CAPP_BUILD_ROOT}/${package}/capp_installed.txt")
endfunction()

#CApp uses these "sentinel" files capp_installed.txt and capp_configured.txt to
#record whether a package has been configured or installed at a particular time.
#In order to see if one of these operations can be treated as a valid cached operation
#then we check whether these files exist and whether they are newer than the CMake
#files that affect the build like app.cmake and the configuration file.
function(capp_sentinel_file_valid)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "PACKAGE;SENTINEL_FILE;RESULT_VARIABLE" "")
  set(package "${arg_PACKAGE}")
  set(sentinel_file "${arg_SENTINEL_FILE}")
  set(result_variable "${arg_RESULT_VARIABLE}")
  if (NOT EXISTS "${sentinel_file}")
    set(${result_variable} FALSE PARENT_SCOPE)
    return()
  endif()
  if (NOT "${sentinel_file}" IS_NEWER_THAN "${CAPP_FLAVOR_ROOT}/flavor.cmake")
    set(${result_variable} FALSE PARENT_SCOPE)
    return()
  endif()
  if (NOT "${sentinel_file}" IS_NEWER_THAN "${CAPP_ROOT}/app.cmake")
    set(${result_variable} FALSE PARENT_SCOPE)
    return()
  endif()
  if (NOT "${sentinel_file}" IS_NEWER_THAN "${CAPP_PACKAGE_ROOT}/${package}/package.cmake")
    set(${result_variable} FALSE PARENT_SCOPE)
    return()
  endif()
  set(${result_variable} TRUE PARENT_SCOPE)
endfunction()

function(capp_dependencies_installed)
  cmake_parse_arguments(PARSE_ARGV 0 capp_dependencies_installed "" "PACKAGE;RESULT_VARIABLE" "")
  set(dependencies_installed TRUE)
  foreach (dependency IN LISTS ${capp_dependencies_installed_PACKAGE}_DEPENDENCIES)
    if (NOT IS_DIRECTORY "${CAPP_PACKAGE_ROOT}/${dependency}")
      message(FATAL_ERROR "${capp_dependencies_installed_PACKAGE} depends on ${dependency}, which is not a package")
    endif()
    if (NOT ${dependency}_IS_INSTALLED)
      set(dependencies_installed FALSE)
    endif()
  endforeach()
  set(${capp_dependencies_installed_RESULT_VARIABLE} ${dependencies_installed} PARENT_SCOPE)
endfunction()

function(capp_initialize_needs)
  foreach(package IN LISTS CAPP_PACKAGES)
    if (IS_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}")
      set(${package}_IS_CLONED TRUE)
    else()
      set(${package}_IS_CLONED FALSE)
    endif()
  endforeach()
  foreach(package IN LISTS CAPP_PACKAGES)
    set(${package}_IS_CONFIGURED TRUE)
    if (NOT ${package}_IS_CLONED)
      set(${package}_IS_CONFIGURED FALSE)
    endif()
    capp_sentinel_file_valid(
      PACKAGE ${package}
      SENTINEL_FILE "${CAPP_BUILD_ROOT}/${package}/capp_configured.txt"
      RESULT_VARIABLE config_sentinel_valid)
    if (NOT config_sentinel_valid)
      set(${package}_IS_CONFIGURED FALSE)
    endif()
    if (NOT ${package}_IS_CONFIGURED)
      capp_invalidate_config(${package})
    endif()
  endforeach()
  foreach(package IN LISTS CAPP_PACKAGES)
    set(${package}_IS_INSTALLED TRUE)
    if (NOT ${package}_IS_CONFIGURED)
      set(${package}_IS_INSTALLED FALSE)
    endif()
    capp_dependencies_installed(
      PACKAGE ${package}
      RESULT_VARIABLE dependencies_installed)
    if (NOT dependencies_installed)
      set(${package}_IS_INSTALLED FALSE)
    endif()
    capp_sentinel_file_valid(
      PACKAGE ${package}
      SENTINEL_FILE "${CAPP_BUILD_ROOT}/${package}/capp_installed.txt"
      RESULT_VARIABLE install_sentinel_valid)
    if (NOT install_sentinel_valid)
      set(${package}_IS_INSTALLED FALSE)
    endif()
    if (NOT ${package}_IS_INSTALLED)
      file(REMOVE "${CAPP_BUILD_ROOT}/${package}/capp_installed.txt")
    endif()
  endforeach()
  foreach(package IN LISTS CAPP_PACKAGES)
    set(${package}_IS_CLONED ${${package}_IS_CLONED} PARENT_SCOPE)
    set(${package}_IS_CONFIGURED ${${package}_IS_CONFIGURED} PARENT_SCOPE)
    set(${package}_IS_INSTALLED ${${package}_IS_INSTALLED} PARENT_SCOPE)
  endforeach()
endfunction()

function(capp_fulfill_needs)
  cmake_parse_arguments(PARSE_ARGV 0 capp_fulfill_needs "" "RESULT_VARIABLE" "BUILD_ARGUMENTS")
  foreach(package IN LISTS CAPP_PACKAGES)
    if (NOT ${package}_IS_CLONED)
      capp_clone(
        PACKAGE ${package}
        RESULT_VARIABLE capp_clone_result
      )
      if (NOT capp_clone_result EQUAL 0)
        message("capp_clone for ${package} failed")
        set(${capp_fulfill_needs_RESULT_VARIABLE} "${capp_clone_result}" PARENT_SCOPE)
        return()
      endif()
      set(${package}_IS_CLONED ${${package}_IS_CLONED} PARENT_SCOPE)
    endif()
    if (NOT ${package}_IS_CONFIGURED)
      capp_configure(
        PACKAGE ${package}
        RESULT_VARIABLE capp_configure_result
      )
      if (NOT capp_configure_result EQUAL 0)
        message("capp_configure for ${package} failed")
        set(${capp_fulfill_needs_RESULT_VARIABLE} "${capp_configure_result}" PARENT_SCOPE)
        return()
      endif()
      set(${package}_IS_CONFIGURED ${${package}_IS_CONFIGURED} PARENT_SCOPE)
    endif()
    if (NOT ${package}_IS_INSTALLED)
      capp_build_install(
        PACKAGE ${package}
        RESULT_VARIABLE capp_build_install_result
        BUILD_ARGUMENTS
        ${capp_fulfill_needs_BUILD_ARGUMENTS}
      )
      if (NOT capp_build_install_result EQUAL 0)
        message("capp_build_install for ${package} failed")
        set(${capp_fulfill_needs_RESULT_VARIABLE} "${capp_build_install_result}" PARENT_SCOPE)
        return()
      endif()
      set(${package}_IS_INSTALLED ${${package}_IS_INSTALLED} PARENT_SCOPE)
    endif()
  endforeach()
  set(${capp_fulfill_needs_RESULT_VARIABLE} 0 PARENT_SCOPE)
endfunction()

function(capp_write_package_file)
  cmake_parse_arguments(PARSE_ARGV 0 capp_write_package_file "" "PACKAGE" "")
  set(file_contents)
  set(file_contents "${file_contents}capp_package(\n")
  set(file_contents "${file_contents}  GIT_URL ${${capp_write_package_file_PACKAGE}_GIT_URL}\n")
  set(file_contents "${file_contents}  COMMIT ${${capp_write_package_file_PACKAGE}_COMMIT}\n")
  set(file_contents "${file_contents}  OPTIONS ${${capp_write_package_file_PACKAGE}_OPTIONS}\n")
  set(file_contents "${file_contents}  DEPENDENCIES ${${capp_write_package_file_PACKAGE}_DEPENDENCIES}\n")
  set(file_contents "${file_contents}  PYTHON_DEPENDENCIES ${${capp_write_package_file_PACKAGE}_PYTHON_DEPENDENCIES}\n")
  set(file_contents "${file_contents})\n")
  set(full_directory "${CAPP_PACKAGE_ROOT}/${capp_write_package_file_PACKAGE}")
  file(MAKE_DIRECTORY "${full_directory}")
  file(WRITE "${full_directory}/package.cmake" "${file_contents}")
endfunction()

function(capp_get_commit)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "GIT_REPO_PATH;COMMIT_VARIABLE;RESULT_VARIABLE" "")
  capp_execute(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${arg_GIT_REPO_PATH}"
    ERROR_QUIET
    RESULT_VARIABLE git_rev_parse_result
    OUTPUT_VARIABLE git_rev_parse_output
    )
  set(${arg_RESULT_VARIABLE} ${git_rev_parse_result} PARENT_SCOPE)
  if (git_rev_parse_result EQUAL 0)
    string(STRIP "${git_rev_parse_output}" git_commit)
    set(${arg_COMMIT_VARIABLE} ${git_commit} PARENT_SCOPE)
  endif()
endfunction()

function(capp_get_package_commit)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "PACKAGE;COMMIT_VARIABLE;RESULT_VARIABLE" "")
  capp_get_commit(
      GIT_REPO_PATH "${CAPP_SOURCE_ROOT}/${arg_PACKAGE}"
      COMMIT_VARIABLE commit
      RESULT_VARIABLE result)
  set(${arg_COMMIT_VARIABLE} ${commit} PARENT_SCOPE)
  set(${arg_RESULT_VARIABLE} ${result} PARENT_SCOPE)
endfunction()

function(capp_get_remote_url)
  cmake_parse_arguments(PARSE_ARGV 0 capp_get_remote_url "" "PACKAGE;REMOTE;GIT_URL_VARIABLE;RESULT_VARIABLE" "")
  capp_execute(
    COMMAND "${GIT_EXECUTABLE}" config --get remote.${capp_get_remote_url_REMOTE}.url
    WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${capp_get_remote_url_PACKAGE}"
    RESULT_VARIABLE git_remote_url_result
    OUTPUT_VARIABLE git_remote_url_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  set(${capp_get_remote_url_RESULT_VARIABLE} ${git_remote_url_result} PARENT_SCOPE)
  if (NOT git_remote_url_result EQUAL 0)
    return()
  endif()
  set(${capp_get_remote_url_GIT_URL_VARIABLE} ${git_remote_url_output} PARENT_SCOPE)
endfunction()

function(capp_init_command)
  cmake_parse_arguments(PARSE_ARGV 0 capp_init_command "" "NAME;RESULT_VARIABLE" "")
  capp_execute(
    COMMAND "${GIT_EXECUTABLE}" init
    WORKING_DIRECTORY "${CAPP_ROOT}"
    RESULT_VARIABLE git_init_result
  )
  if (NOT git_init_result EQUAL 0)
    set(${capp_init_command_RESULT_VARIABLE} ${git_init_result} PARENT_SCOPE)
    return()
  endif()
  file(WRITE "${CAPP_ROOT}/app.cmake" "capp_app(\n  ROOT_PACKAGES\n  )")
  capp_add_file(
    FILE "${CAPP_ROOT}/app.cmake"
    RESULT_VARIABLE capp_add_file_result
  )
  if (NOT capp_add_file_result EQUAL 0)
    set(${capp_init_command_RESULT_VARIABLE} ${capp_add_file_result} PARENT_SCOPE)
    return()
  endif()
  file(WRITE "${CAPP_ROOT}/.gitignore" "source\nflavor/*/build\nflavor/*/install\nflavor/*/venv")
  capp_add_file(
    FILE "${CAPP_ROOT}/.gitignore"
    RESULT_VARIABLE capp_add_file_result
  )
  if (NOT capp_add_file_result EQUAL 0)
    set(${capp_init_command_RESULT_VARIABLE} ${capp_add_file_result} PARENT_SCOPE)
    return()
  endif()
  foreach(filename "capp.cmake" "capp-setup.sh")
    capp_add_file(
      FILE "${CAPP_ROOT}/${filename}"
      RESULT_VARIABLE capp_add_file_result
    )
    if (NOT capp_add_file_result EQUAL 0)
      set(${capp_init_command_RESULT_VARIABLE} ${capp_add_file_result} PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${capp_init_command_RESULT_VARIABLE} 0 PARENT_SCOPE)
endfunction()

function(capp_clone_command)
  cmake_parse_arguments(PARSE_ARGV 0 capp_clone_command "" "RESULT_VARIABLE" "GIT_ARGUMENTS")
  file(MAKE_DIRECTORY "${CAPP_SOURCE_ROOT}")
  capp_execute(
    COMMAND "${GIT_EXECUTABLE}" clone --depth 1 --recursive
            ${capp_clone_command_GIT_ARGUMENTS}
    WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}"
    RESULT_VARIABLE git_clone_result
    ERROR_VARIABLE git_clone_error
    OUTPUT_VARIABLE git_clone_output
  )
  if (NOT git_clone_result EQUAL 0)
    capp_list_to_string(LIST capp_clone_command_GIT_ARGUMENTS STRING arg_string)
    message("CApp: git clone --depth 1 --recursive ${arg_string} failed:\n${git_clone_output}\n${git_clone_error}")
    set(${capp_clone_command_RESULT_VARIABLE} ${git_clone_result} PARENT_SCOPE)
    return()
  endif()
  string(REGEX MATCH "'[^']+'" git_directory_quoted "${git_clone_error}")
  string(LENGTH "${git_directory_quoted}" git_directory_quoted_length)
  math(EXPR git_directory_length "${git_directory_quoted_length} - 2")
  string(SUBSTRING "${git_directory_quoted}" 1 ${git_directory_length} package)
  capp_get_remote_url(
    PACKAGE ${package}
    REMOTE origin
    GIT_URL_VARIABLE ${package}_GIT_URL
    RESULT_VARIABLE capp_get_remote_url_result
  )
  if (NOT capp_get_remote_url_result EQUAL 0)
    set(${capp_clone_command_RESULT_VARIABLE} ${capp_get_remote_url_result} PARENT_SCOPE)
    return()
  endif()
  capp_get_package_commit(
    PACKAGE ${package}
    COMMIT_VARIABLE ${package}_COMMIT
    RESULT_VARIABLE capp_get_commit_result
  )
  if (NOT capp_get_commit_result EQUAL 0)
    set(${capp_clone_command_RESULT_VARIABLE} ${capp_get_commit_result} PARENT_SCOPE)
    return()
  endif()
  capp_write_package_file(PACKAGE ${package})
  capp_add_file(
    FILE "${CAPP_PACKAGE_ROOT}/${package}/package.cmake"
    RESULT_VARIABLE capp_add_file_result
  )
  if (NOT capp_add_file_result EQUAL 0)
    set(${capp_clone_command_RESULT_VARIABLE} ${capp_add_file_result} PARENT_SCOPE)
    return()
  endif()
  set(${capp_clone_command_RESULT_VARIABLE} 0 PARENT_SCOPE)
endfunction()

function(capp_commit_command)
  cmake_parse_arguments(PARSE_ARGV 0 capp_commit_command "" "PACKAGE;RESULT_VARIABLE" "")
  if (${capp_commit_command_PACKAGE}_IS_LOCAL)
    # Local packages are skipped by the commit command
    set(${capp_commit_command_RESULT_VARIABLE} 0 PARENT_SCOPE)
    return()
  endif()
  #Safety check: make sure all the user's changes are committed before proceeding.
  capp_changes_committed(
    PACKAGE ${capp_commit_command_PACKAGE}
    RESULT_VARIABLE uncommitted_result
    OUTPUT_VARIABLE uncommitted_output)
  if (NOT uncommitted_result EQUAL 0)
    message("\nCApp refusing to commit ${capp_commit_command_PACKAGE} because it has uncommitted changes:\n${uncommitted_output}")
    set(${capp_commit_command_RESULT_VARIABLE} -1 PARENT_SCOPE)
    return()
  endif()
  capp_get_package_commit(
    PACKAGE ${capp_commit_command_PACKAGE}
    COMMIT_VARIABLE new_commit
    RESULT_VARIABLE capp_get_commit_result
  )
  if ("${new_commit}" STREQUAL "${${capp_commit_command_PACKAGE}_COMMIT}")
    message("\nCApp: commit for ${capp_commit_command_PACKAGE} has not changed")
    set(${capp_commit_command_RESULT_VARIABLE} 0 PARENT_SCOPE)
    return()
  endif()
  if (NOT capp_get_commit_result EQUAL 0)
    message("CApp commit command could not get the current commit for ${capp_commit_command_PACKAGE}")
    set(${capp_commit_command_RESULT_VARIABLE} ${capp_get_commit_result} PARENT_SCOPE)
    return()
  endif()
  #All the following work is just to figure out what Git URL to use,
  #which we assume is the URL for one of the remotes.
  set(remote)
  #First, we pursue a common case of a package we keep up closely with that
  #is checked out to a branch which has an upstream branch defined.
  #If that is the case, the following command will print the ref of the upstream
  #branch.
  capp_execute(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --symbolic-full-name @{u}
    WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${capp_commit_command_PACKAGE}"
    RESULT_VARIABLE upstream_ref_result
    ERROR_VARIABLE upstream_ref_error
    OUTPUT_VARIABLE upstream_ref
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (upstream_ref_result EQUAL 0)
    #If we are here then there is a clear upstream ref.
    message("CApp: for ${capp_commit_command_PACKAGE}, there is a clear upstream ref ${upstream_ref}")
    string(REGEX REPLACE "${CAPP_REMOTE_REF_REGEX}" "\\1" remote "${upstream_ref}")
    string(REGEX REPLACE "${CAPP_REMOTE_REF_REGEX}" "\\2" branch "${upstream_ref}")
    #Now, it's still important that we check whether the current commit has been pushed
    #to the clear upstream branch in this case.
    capp_execute(
      COMMAND "${GIT_EXECUTABLE}" merge-base --is-ancestor ${new_commit} ${remote}/${branch}
      WORKING_DIRECTORY  "${CAPP_SOURCE_ROOT}/${capp_commit_command_PACKAGE}"
      RESULT_VARIABLE is_ancestor_result)
    if (NOT is_ancestor_result EQUAL 0)
      message("\nCApp refusing to commit ${capp_commit_command_PACKAGE} because the current commit ${new_commit} is not pushed to the upstream branch ${branch} on remote ${remote}\n")
      set(${capp_commit_command_RESULT_VARIABLE} -1 PARENT_SCOPE)
      return()
    endif()
    message("CApp: branch ${branch} on remote ${remote} contains commit ${new_commit}")
  endif()
  if (NOT remote)
    #Now comes the harder case... we are checking out a "detached HEAD" commit
    #of some repository, so there is no current branch let alone an upstream branch.
    #Fortunately, Git still is able to tell us at least whether the current commit
    #is pushed to any remote ref, and we can go from there.
    #The following command will print lines that contain remotes that contain the commit:
    capp_execute(
      COMMAND "${GIT_EXECUTABLE}" for-each-ref --contains ${new_commit}
      WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${capp_commit_command_PACKAGE}"
      RESULT_VARIABLE containing_refs_result
      ERROR_VARIABLE containing_refs_error
      OUTPUT_VARIABLE containing_refs_output)
    if (NOT containing_refs_result EQUAL 0)
      message("for ${capp_commit_command_PACKAGE}, git for-each-ref --contains ${new_commit} failed")
      set(${capp_commit_command_RESULT_VARIABLE} ${containing_refs_result} PARENT_SCOPE)
      return()
    endif()
    #MATCHALL will create a CMake list with all the remote refs that contain the commit
    string(REGEX MATCHALL "${CAPP_REMOTE_REF_REGEX}" containing_remote_refs "${containing_refs_output}")
    if (NOT containing_remote_refs)
      #If there is nothing here, then there really is zero evidence that the current commit
      #has ever left this machine and made it onto a Git host server.
      #In this case we assume it is the common user error of not pushing their commits.
      message("\nCApp refusing to commit ${capp_commit_command_PACKAGE} because commit ${new_commit} is not pushed to any remote!\n")
      set(${capp_commit_command_RESULT_VARIABLE} -1 PARENT_SCOPE)
      return()
    endif()
    #If we are here, then we did find some remote refs that contain the current commit
    #The following code creates a list of the remotes that those refs are in
    set(containing_remotes)
    foreach(containing_remote_ref IN LISTS containing_remote_refs)
      #Use the subexpressions built into the remote ref regex to extract the remote
      string(REGEX REPLACE "${CAPP_REMOTE_REF_REGEX}" "\\1" containing_remote "${containing_remote_ref}")
      list(APPEND containing_remotes ${containing_remote})
    endforeach()
    list(REMOVE_DUPLICATES containing_remotes)
    message("CApp: for ${capp_commit_command_PACKAGE}, containing_remotes=${containing_remotes}")
    #Now we need to pick one of those remotes as the official URL.
    #all we know is origin is special, of if origin contains it then
    #let's pick that one.
    if ("origin" IN_LIST containing_remotes)
      set(remote "origin")
    else()
      #Otherwise, let's just pick the first remote that contains this commit
      list(GET containing_remotes 0 remote)
    endif()
  endif()
  message("CApp: chose remote ${remote} to commit ${capp_commit_command_PACKAGE}")
  capp_get_remote_url(
    PACKAGE ${capp_commit_command_PACKAGE}
    REMOTE ${remote}
    GIT_URL_VARIABLE new_git_url
    RESULT_VARIABLE get_remote_url_result
    )
  if (NOT get_remote_url_result EQUAL 0)
    message("failed to get the Git URL for remote ${remote} of ${capp_commit_command_PACKAGE}\n")
    set(${capp_commit_command_RESULT_VARIABLE} "${get_remote_url_result}" PARENT_SCOPE)
    return()
  endif()
  file(READ "${CAPP_PACKAGE_ROOT}/${capp_commit_command_PACKAGE}/package.cmake" old_package_contents)
  string(REGEX REPLACE "COMMIT [a-z0-9]+" "COMMIT ${new_commit}" commit_package_contents "${old_package_contents}")
  string(REGEX REPLACE "\n([ \r\t]*)GIT_URL [_a-zA-Z0-9:@/\\.-]+" "\n\\1GIT_URL ${new_git_url}" new_package_contents "${commit_package_contents}")
  # if the file doesn't change, don't write to it because that will trigger reconfiguration later
  if (NOT old_package_contents STREQUAL new_package_contents)
    file(WRITE "${CAPP_PACKAGE_ROOT}/${capp_commit_command_PACKAGE}/package.cmake" "${new_package_contents}")
  endif()
  set(${capp_commit_command_RESULT_VARIABLE} 0 PARENT_SCOPE)
  capp_add_file(
    FILE "${CAPP_PACKAGE_ROOT}/${capp_commit_command_PACKAGE}/package.cmake"
    RESULT_VARIABLE capp_add_file_result
  )
  if (NOT capp_add_file_result EQUAL 0)
    set(${capp_commit_command_RESULT_VARIABLE} ${capp_add_file_result} PARENT_SCOPE)
    return()
  endif()
  set(${capp_commit_command_RESULT_VARIABLE} 0 PARENT_SCOPE)
endfunction()

function(capp_checkout_command)
  cmake_parse_arguments(PARSE_ARGV 0 capp_checkout_command "" "RESULT_VARIABLE" "PACKAGES")
  foreach(package IN LISTS capp_checkout_command_PACKAGES)
    if (EXISTS "${CAPP_SOURCE_ROOT}/${package}")
      if (NOT ${${package}_IS_LOCAL})
        capp_checkout(
          PACKAGE ${package}
          RESULT_VARIABLE checkout_result)
        if (NOT checkout_result EQUAL 0)
          set(${capp_checkout_command_RESULT_VARIABLE} ${checkout_result} PARENT_SCOPE)
          return()
        endif()
      endif()
    else()
      if (${${package}_IS_LOCAL})
        message("CApp: ${package} has IS_LOCAL but ${CAPP_SOURCE_ROOT}/${package} does not exist")
        set(${capp_checkout_command_RESULT_VARIABLE} -1 PARENT_SCOPE)
        return()
      endif()
      capp_clone(
        PACKAGE ${package}
        RESULT_VARIABLE clone_result)
      if (NOT clone_result EQUAL 0)
        set(${capp_checkout_command_RESULT_VARIABLE} ${clone_result} PARENT_SCOPE)
        return()
      endif()
    endif()
  endforeach()
  set(${capp_checkout_command_RESULT_VARIABLE} 0 PARENT_SCOPE)
endfunction()

function(capp_develop_command)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "RESULT_VARIABLE" "PACKAGES")
  foreach(package IN LISTS arg_PACKAGES)
    capp_execute(
      COMMAND "${GIT_EXECUTABLE}" fetch --unshallow
      WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
      RESULT_VARIABLE result)
    if (NOT result EQUAL 0)
      message("CApp: git fetch --unshallow failed in ${package}")
      set(${arg_RESULT_VARIABLE} "${result}" PARENT_SCOPE)
      return()
    endif()
    capp_execute(
      COMMAND "${GIT_EXECUTABLE}" remote set-branches origin *
      WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
      RESULT_VARIABLE result)
    if (NOT result EQUAL 0)
      message("CApp: git remote set-branches origin '*' failed in ${package}")
      set(${arg_RESULT_VARIABLE} "${result}" PARENT_SCOPE)
      return()
    endif()
    capp_execute(
      COMMAND "${GIT_EXECUTABLE}" fetch origin
      WORKING_DIRECTORY "${CAPP_SOURCE_ROOT}/${package}"
      RESULT_VARIABLE result)
    if (NOT result EQUAL 0)
      message("CApp: git fetch origin failed in ${package}")
      set(${arg_RESULT_VARIABLE} "${result}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${arg_RESULT_VARIABLE} 0 PARENT_SCOPE)
endfunction()

function(capp_pull_command)
  cmake_parse_arguments(PARSE_ARGV 0 capp_pull_command "" "RESULT_VARIABLE" "")
  message("\nCApp is pulling the build repository\n")
  capp_execute(
    COMMAND "${GIT_EXECUTABLE}" pull
    WORKING_DIRECTORY "${CAPP_ROOT}"
    RESULT_VARIABLE pull_result
    )
  if (NOT pull_result EQUAL 0)
    set(${capp_pull_command_RESULT_VARIABLE} "${pull_result}" PARENT_SCOPE)
    return()
  endif()
  capp_read_package_files_by_dependency()
  capp_topsort_packages()
  capp_checkout_command(RESULT_VARIABLE capp_checkout_result PACKAGES ${CAPP_PACKAGES})
  set(${capp_pull_command_RESULT_VARIABLE} "${capp_checkout_result}" PARENT_SCOPE)
endfunction()

function(capp_test_command)
  cmake_parse_arguments(PARSE_ARGV 0 capp_test_command "" "RESULT_VARIABLE" "PACKAGES;ARGUMENTS")
  foreach(package IN LISTS capp_test_command_PACKAGES)
    capp_execute(
      WORKING_DIRECTORY "${CAPP_BUILD_ROOT}/${package}"
      RESULT_VARIABLE package_test_result
      COMMAND ctest ${capp_test_command_ARGUMENTS} -C ${${package}_BUILD_TYPE})
    if (NOT package_test_result EQUAL 0)
      if (EXISTS "${CAPP_BUILD_ROOT}/${package}")
        message("CApp: running CTest in ${CAPP_BUILD_ROOT}/${package} failed")
      else()
        message("CApp: build directory ${CAPP_BUILD_ROOT}/${package} does not exist! Usually this means that the package ${package} has not been compiled for the flavor ${CAPP_FLAVOR} yet")
      endif()
      set(${capp_test_command_RESULT_VARIABLE} "${package_test_result}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${capp_test_command_RESULT_VARIABLE} 0 PARENT_SCOPE)
endfunction()

function(capp_install_command)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "RESULT_VARIABLE;PREFIX" "PACKAGES")
  if (NOT arg_PREFIX)
    message("CApp install was not given a prefix, please use --prefix /your/path")
    set(${arg_RESULT_VARIABLE} -1 PARENT_SCOPE)
    return()
  endif()
  foreach(package IN LISTS arg_PACKAGES)
    capp_get_subdirectories(package_subdirs "${CAPP_INSTALL_ROOT}/${package}")
    foreach(package_subdir IN LISTS package_subdirs)
      message("CApp copying ${CAPP_INSTALL_ROOT}/${package}/${package_subdir} to ${arg_PREFIX}")
      file(INSTALL "${CAPP_INSTALL_ROOT}/${package}/${package_subdir}"
           DESTINATION "${arg_PREFIX}"
           USE_SOURCE_PERMISSIONS)
    endforeach()
  endforeach()
  set(${arg_RESULT_VARIABLE} 0 PARENT_SCOPE)
endfunction()

function(capp_export_command)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "RESULT_VARIABLE" "PACKAGES")
  set(export_content "[\n")
  list(LENGTH arg_PACKAGES package_count)
  math(EXPR last_package_index "${package_count} - 1")
  foreach(package_index RANGE ${last_package_index})
    list(GET arg_PACKAGES ${package_index} package)
    string(APPEND export_content "  {\n")
    string(APPEND export_content "    \"name\" : \"${package}\",\n")
    string(APPEND export_content "    \"git\" : \"${${package}_GIT_URL}\",\n")
    string(APPEND export_content "    \"commit\" : \"${${package}_COMMIT}\",\n")
    if (${package}_HAS_SUBMODULES)
      set(has_submodules "true")
    else()
      set(has_submodules "false")
    endif()
    string(APPEND export_content "    \"submodules\" : ${has_submodules},\n")
    string(APPEND export_content "    \"dependencies\" : [\n")
    set(dependencies "${${package}_DEPENDENCIES}")
    if (dependencies)
      list(LENGTH dependencies dependency_count)
      math(EXPR last_dependency_index "${dependency_count} - 1")
      foreach(dependency_index RANGE ${last_dependency_index})
        list(GET dependencies ${dependency_index} dependency)
        string(APPEND export_content "      \"${dependency}\"")
        if (dependency_index EQUAL last_dependency_index)
          string(APPEND export_content "\n")
        else()
          string(APPEND export_content ",\n")
        endif()
      endforeach()
    endif()
    string(APPEND export_content "    ],\n")
    string(APPEND export_content "    \"options\" : [\n")
    set(options "${${package}_OPTIONS}")
    if (options)
      list(LENGTH options option_count)
      math(EXPR last_option_index "${option_count} - 1")
      foreach(option_index RANGE ${last_option_index})
        list(GET options ${option_index} option)
        string(APPEND export_content "      \"${option}\"")
        if (option_index EQUAL last_option_index)
          string(APPEND export_content "\n")
        else()
          string(APPEND export_content ",\n")
        endif()
      endforeach()
    endif()
    string(APPEND export_content "    ]\n")
    if (package_index EQUAL last_package_index)
      string(APPEND export_content "  }\n")
    else()
      string(APPEND export_content "  },\n")
    endif()
  endforeach()
  string(APPEND export_content "]\n")
  file(WRITE "capp.json" "${export_content}")
  set(${arg_RESULT_VARIABLE} 0 PARENT_SCOPE)
endfunction()

function(capp_environment_command)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "RESULT_VARIABLE;MODE" "")
  set(path "")
  set(pythonpath "")
  set(venv_path "${CAPP_VENV_ROOT}/bin")
  if (EXISTS "${venv_path}")
    list(APPEND path "${venv_path}")
  endif()
  foreach(package IN LISTS CAPP_PACKAGES)
    set(package_path "${CAPP_INSTALL_ROOT}/${package}/bin")
    if (EXISTS "${package_path}")
      list(APPEND path "${package_path}")
    endif()
    foreach(relpath IN LISTS ${package}_PYTHONPATH)
      set(package_pythonpath "${CAPP_INSTALL_ROOT}/${package}/${relpath}")
      if (EXISTS "${package_pythonpath}")
        list(APPEND pythonpath "${package_pythonpath}")
      endif()
    endforeach()
  endforeach()
  if (arg_MODE STREQUAL "load")
    capp_add_paths(newpath PATH "${path}")
    capp_stdout("export PATH=\"${newpath}\"")
    capp_add_paths(newpath PYTHONPATH "${pythonpath}")
    capp_stdout("export PYTHONPATH=\"${newpath}\"")
  elseif (arg_MODE STREQUAL "unload")
    capp_remove_paths(newpath PATH "${path}")
    capp_stdout("export PATH=\"${newpath}\"")
    capp_remove_paths(newpath PYTHONPATH "${pythonpath}")
    capp_stdout("export PYTHONPATH=\"${newpath}\"")
  elseif(arg_MODE STREQUAL "lmod")
    foreach(entry IN LISTS path)
      capp_stdout("prepend_path(\"PATH\", \"${entry}\")")
    endforeach()
    foreach(entry IN LISTS pythonpath)
      capp_stdout("prepend_path(\"PYTHONPATH\", \"${entry}\")")
    endforeach()
  endif()
  set(${arg_RESULT_VARIABLE} 0 PARENT_SCOPE)
endfunction()

function(capp_separate_command_args)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "PACKAGES_VARIABLE;BUILD_ARGUMENTS_VARIABLE;TEST_ARGUMENTS_VARIABLE;INSTALL_ARGUMENTS_VARIABLE" "INPUT_ARGUMENTS")
  set(build_args)
  set(test_args)
  set(install_args)
  set(packages)
  while (arg_INPUT_ARGUMENTS)
    list(POP_FRONT arg_INPUT_ARGUMENTS arg)
    list(FIND CAPP_PACKAGES "${arg}" package_index)
    if (arg STREQUAL "--parallel" OR arg STREQUAL "-j")
      list(POP_FRONT arg_INPUT_ARGUMENTS n)
      list(APPEND build_args "${arg}" "${n}")
      list(APPEND test_args "${arg}" "${n}")
    elseif (arg MATCHES "-j[0-9]+")
      string(SUBSTRING "${arg}" 2 -1 n)
      list(APPEND build_args -j ${n})
      list(APPEND test_args -j ${n})
    elseif (arg STREQUAL "--verbose")
      list(APPEND build_args "${arg}")
      list(APPEND test_args "${arg}")
    elseif (arg STREQUAL "-v")
      list(APPEND build_args "${arg}")
    elseif (arg STREQUAL "-V")
      list(APPEND test_args "${arg}")
    elseif (arg STREQUAL "-R")
      list(POP_FRONT arg_INPUT_ARGUMENTS regex)
      list(APPEND test_args -R "${regex}")
    elseif (arg STREQUAL "--prefix")
      list(POP_FRONT arg_INPUT_ARGUMENTS prefix)
      list(APPEND install_args PREFIX "${prefix}")
    elseif (NOT package_index EQUAL -1)
      list(APPEND packages "${arg}")
    endif()
  endwhile()
  if (NOT packages)
    # If the user didn't list packages as command line arguments
    # but they are running inside a source/<package> directory,
    # operate only on <package>
    set(source_dir_regex "${CAPP_SOURCE_ROOT}/([^/])")
    if (CMAKE_CURRENT_SOURCE_DIR MATCHES "${source_dir_regex}")
      string(REGEX REPLACE "${source_dir_regex}" "\\1" package "${CMAKE_CURRENT_SOURCE_DIR}")
      set(packages "${package}")
    else()
      set(packages "${CAPP_PACKAGES}")
    endif()
  endif()
  if (arg_PACKAGES_VARIABLE)
    set(${arg_PACKAGES_VARIABLE} "${packages}" PARENT_SCOPE)
  endif()
  if (arg_BUILD_ARGUMENTS_VARIABLE)
    set(${arg_BUILD_ARGUMENTS_VARIABLE} "${build_args}" PARENT_SCOPE)
  endif()
  if (arg_TEST_ARGUMENTS_VARIABLE)
    set(${arg_TEST_ARGUMENTS_VARIABLE} "${test_args}" PARENT_SCOPE)
  endif()
  if (arg_INSTALL_ARGUMENTS_VARIABLE)
    set(${arg_INSTALL_ARGUMENTS_VARIABLE} "${install_args}" PARENT_SCOPE)
  endif()
endfunction()

function(capp_parse_main_args)
  set(remaining_args "${CAPP_CMDLINE_ARGS}")
  set(definition_regex "-D([^=]+)=([^=]*)")
  set(flavor_regex "(-f)|(--flavor)")
  set(command_args)
  while (remaining_args)
    list(POP_FRONT remaining_args arg)
    if (arg STREQUAL "--")
    elseif (arg STREQUAL "")
    elseif (NOT command)
      set(command "${arg}")
    elseif (arg MATCHES "${definition_regex}")
      string(REGEX REPLACE "${definition_regex}" "\\1" variable_name "${arg}")
      string(REGEX REPLACE "${definition_regex}" "\\2" variable_value "${arg}")
      set("${variable_name}" "${variable_value}" PARENT_SCOPE)
    elseif (arg MATCHES "${flavor_regex}")
      list(POP_FRONT remaining_args flavor)
    elseif (arg STREQUAL "--proxy")
      list(POP_FRONT remaining_args proxy)
    elseif (arg STREQUAL "--prefix")
      list(POP_FRONT remaining_args prefix)
    else()
      list(APPEND command_args "${arg}")
    endif()
  endwhile()
  if (NOT command)
    message(FATAL_ERROR "No command specified!")
    return()
  endif()
  set(CAPP_COMMAND "${command}" PARENT_SCOPE)
  set(CAPP_COMMAND_ARGUMENTS "${command_args}" PARENT_SCOPE)
  set(CAPP_FLAVOR "${flavor}" PARENT_SCOPE)
  set(CAPP_PROXY "${proxy}" PARENT_SCOPE)
  set(pip_flags
      --trusted-host pypi.python.org
      --trusted-host files.pythonhosted.org
      )
  if (proxy)
    list(APPEND pip_flags --proxy ${proxy})
  endif()
  if (prefix)
    set(CAPP_PREFIX "${prefix}" PARENT_SCOPE)
  endif()
  set(CAPP_PIP_FLAGS "${pip_flags}" PARENT_SCOPE)
endfunction()

capp_parse_main_args()
set(CAPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
if (CAPP_COMMAND STREQUAL "init")
  capp_init_command(
    NAME ${CAPP_COMMAND_ARGUMENTS}
    RESULT_VARIABLE capp_command_result
  )
elseif (CAPP_COMMAND STREQUAL "clone")
  capp_find_root()
  capp_clone_command(
    GIT_ARGUMENTS ${CAPP_COMMAND_ARGUMENTS}
    RESULT_VARIABLE capp_command_result
  )
elseif(CAPP_COMMAND STREQUAL "build")
  capp_find_root()
  capp_setup_flavor()
  capp_read_package_files_by_dependency()
  capp_topsort_packages()
  capp_separate_command_args(
    INPUT_ARGUMENTS ${CAPP_COMMAND_ARGUMENTS}
    PACKAGES_VARIABLE package_list
    BUILD_ARGUMENTS_VARIABLE build_args)
  capp_initialize_needs()
  capp_fulfill_needs(
    RESULT_VARIABLE capp_command_result
    BUILD_ARGUMENTS ${build_args}
  )
elseif(CAPP_COMMAND STREQUAL "rebuild")
  capp_find_root()
  capp_setup_flavor()
  capp_read_package_files_by_dependency()
  capp_topsort_packages()
  capp_separate_command_args(
    INPUT_ARGUMENTS ${CAPP_COMMAND_ARGUMENTS}
    PACKAGES_VARIABLE build_list
    BUILD_ARGUMENTS_VARIABLE build_args)
  foreach (package IN LISTS build_list)
    capp_invalidate_install(${package})
  endforeach()
  capp_initialize_needs()
  capp_fulfill_needs(
    RESULT_VARIABLE capp_command_result
    BUILD_ARGUMENTS
    ${build_args}
  )
elseif(CAPP_COMMAND STREQUAL "reconfig")
  capp_find_root()
  capp_setup_flavor()
  capp_read_package_files_by_dependency()
  capp_topsort_packages()
  capp_separate_command_args(
    INPUT_ARGUMENTS ${CAPP_COMMAND_ARGUMENTS}
    PACKAGES_VARIABLE config_list
    BUILD_ARGUMENTS_VARIABLE build_args)
  foreach (package IN LISTS config_list)
    capp_invalidate_config(${package})
  endforeach()
  capp_initialize_needs()
  capp_fulfill_needs(
    RESULT_VARIABLE capp_command_result
    BUILD_ARGUMENTS
    ${build_args}
  )
elseif(CAPP_COMMAND STREQUAL "test")
  capp_find_root()
  capp_setup_flavor()
  capp_read_package_files_by_dependency()
  capp_topsort_packages()
  capp_separate_command_args(
    INPUT_ARGUMENTS ${CAPP_COMMAND_ARGUMENTS}
    TEST_ARGUMENTS_VARIABLE test_args
    PACKAGES_VARIABLE test_list)
  capp_test_command(
    ARGUMENTS ${test_args}
    PACKAGES ${test_list}
    RESULT_VARIABLE capp_command_result)
elseif(CAPP_COMMAND STREQUAL "install")
  capp_find_root()
  capp_setup_flavor()
  capp_read_package_files_by_dependency()
  capp_topsort_packages()
  capp_separate_command_args(
    INPUT_ARGUMENTS ${CAPP_COMMAND_ARGUMENTS}
    PACKAGES_VARIABLE package_list
    BUILD_ARGUMENTS_VARIABLE build_args
    INSTALL_ARGUMENTS_VARIABLE install_args)
  capp_initialize_needs()
  capp_fulfill_needs(
    RESULT_VARIABLE capp_command_result
    BUILD_ARGUMENTS ${build_args}
  )
  capp_install_command(
    PACKAGES ${package_list}
    ${install_args}
    RESULT_VARIABLE capp_command_result)
elseif(CAPP_COMMAND STREQUAL "commit")
  capp_find_root()
  capp_setup_flavor(OPTIONAL)
  if (CAPP_FLAVOR)
    capp_read_package_files_by_dependency()
    capp_topsort_packages()
  else()
    capp_read_all_package_files()
  endif()
  capp_separate_command_args(
    INPUT_ARGUMENTS ${CAPP_COMMAND_ARGUMENTS}
    PACKAGES_VARIABLE commit_list)
  set(capp_command_result 0)
  foreach (package IN LISTS commit_list)
    if (EXISTS "${CAPP_SOURCE_ROOT}/${package}")
      capp_commit_command(
        PACKAGE ${package}
        RESULT_VARIABLE capp_command_result
      )
      if (NOT capp_command_result EQUAL 0)
        break()
      endif()
    elseif (NOT EXISTS "${CAPP_PACKAGE_ROOT}/${package}")
      message("CApp commit: ${package} is not checked out")
      set(capp_command_result -1)
    endif()
  endforeach()
elseif(CAPP_COMMAND STREQUAL "checkout")
  capp_find_root()
  capp_setup_flavor(OPTIONAL)
  if (CAPP_FLAVOR)
    capp_read_package_files_by_dependency()
    capp_topsort_packages()
  else()
    capp_read_all_package_files()
  endif()
  capp_separate_command_args(
    INPUT_ARGUMENTS ${CAPP_COMMAND_ARGUMENTS}
    PACKAGES_VARIABLE checkout_list)
  capp_checkout_command(
    RESULT_VARIABLE capp_command_result
    PACKAGES ${checkout_list}
  )
elseif(CAPP_COMMAND STREQUAL "develop")
  capp_find_root()
  capp_setup_flavor(OPTIONAL)
  if (CAPP_FLAVOR)
    capp_read_package_files_by_dependency()
    capp_topsort_packages()
  else()
    capp_read_all_package_files()
  endif()
  capp_separate_command_args(
    INPUT_ARGUMENTS ${CAPP_COMMAND_ARGUMENTS}
    PACKAGES_VARIABLE develop_list)
  if (NOT develop_list)
    message(FATAL_ERROR "CApp: develop command expects one or more package names")
  endif()
  capp_develop_command(
    RESULT_VARIABLE capp_command_result
    PACKAGES "${develop_list}"
  )
elseif(CAPP_COMMAND STREQUAL "pull")
  capp_find_root()
  capp_pull_command(
    RESULT_VARIABLE capp_command_result
  )
elseif(CAPP_COMMAND STREQUAL "export")
  capp_find_root()
  capp_setup_flavor(OPTIONAL)
  if (CAPP_FLAVOR)
    capp_read_package_files_by_dependency()
    capp_topsort_packages()
  else()
    capp_read_all_package_files()
  endif()
  capp_separate_command_args(
    INPUT_ARGUMENTS ${CAPP_COMMAND_ARGUMENTS}
    PACKAGES_VARIABLE checkout_list)
  capp_export_command(
    RESULT_VARIABLE capp_command_result
    PACKAGES ${checkout_list}
  )
elseif(CAPP_COMMAND MATCHES "load|unload|lmod")
  capp_find_root()
  capp_setup_flavor()
  capp_read_package_files_by_dependency()
  capp_topsort_packages()
  capp_environment_command(
    RESULT_VARIABLE capp_command_result
    MODE ${CAPP_COMMAND}
  )
elseif(CAPP_COMMAND STREQUAL "clean")
  capp_find_root()
  capp_setup_flavor()
  message("CApp: removing ${CAPP_BUILD_ROOT}, ${CAPP_INSTALL_ROOT}, and ${CAPP_VENV_ROOT}")
  file(REMOVE_RECURSE "${CAPP_BUILD_ROOT}")
  file(REMOVE_RECURSE "${CAPP_INSTALL_ROOT}")
  file(REMOVE_RECURSE "${CAPP_VENV_ROOT}")
  set(capp_command_result 0)
else()
  message(FATAL_ERROR "Unknown command ${CAPP_COMMAND}!")
endif()

if (NOT capp_command_result EQUAL 0)
  message(FATAL_ERROR "CApp command ${CAPP_COMMAND} failed")
endif()
