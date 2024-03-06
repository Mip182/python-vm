"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import textwrap
import types
import typing as tp
from sys import exc_info

CO_VARARGS = 4
CO_VARKEYWORDS = 8

ERR_TOO_MANY_POS_ARGS = "Too many positional arguments"
ERR_TOO_MANY_KW_ARGS = "Too many keyword arguments"
ERR_MULT_VALUES_FOR_ARG = "Multiple values for arguments"
ERR_MISSING_POS_ARGS = "Missing positional arguments"
ERR_MISSING_KWONLY_ARGS = "Missing keyword-only arguments"
ERR_POSONLY_PASSED_AS_KW = "Positional-only argument passed as keyword argument"

DEBUG = 0


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.11/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    def __init__(
        self,
        frame_code: types.CodeType,
        frame_builtins: dict[str, tp.Any],
        frame_globals: dict[str, tp.Any],
        frame_locals: dict[str, tp.Any],
    ) -> None:
        self.is_end = False
        self.kw_names = None
        self.f_lasti = None
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.current_arg = None
        self.locals["textwrap"] = textwrap
        self.data_stack: tp.Any = []
        self.offset_to_line = {}  # type: ignore
        self.current_line = 0
        self.lines_of_code = []  # type: ignore
        self.lengths_of_instr = []  # type: ignore
        self.return_value = None

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        if len(self.data_stack) > 0:
            return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def copy_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        le = []
        for i in range(0, n):
            le.append(self.pop())
        for key in reversed(le):
            self.push(key)
        self.push(le[n - 1])

    def list_to_tuple_op(self, n: tp.Any) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        li = self.pop()
        self.push(tuple(li))

    def swap_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        le = []
        for i in range(0, n):
            le.append(self.pop())
        self.push(le[0])
        for i in range(1, n - 1):
            self.push(le[n - 1 - i])
        self.push(le[n - 1])

    def run(self) -> tp.Any:
        cur_line = 0
        instr_names = [
            # "for_iter",
            # "pop_jump_forward_if_true",
            # "pop_jump_backward_if_true",
            # "pop_jump_forward_if_none",
            # "jump_if_false_or_pop",
            # "jump_backward",
            # "jump_forward",
            # "jump_if_true_or_pop_op",
            # "jump_if_false_or_pop_op",
            # "kw_names",
            # "make_function"
        ]
        for instruction in dis.get_instructions(self.code):
            if DEBUG:
                print(
                    instruction.offset,
                    instruction.opname.lower() + "_op",
                    instruction.argval,
                )
                instr_names.append(instruction.opname.lower() + "_op")
            self.lines_of_code.append(
                (
                    getattr(self, instruction.opname.lower() + "_op"),
                    instruction.argval,
                    instruction.arg,
                )
            )
            self.offset_to_line[instruction.offset] = cur_line
            cur_line += 1
        while not self.is_end and self.current_line < len(self.lines_of_code):
            if DEBUG:
                print(
                    "cur_line ",
                    self.current_line,
                    instr_names[self.current_line],
                    self.data_stack,
                )
            self.current_arg = self.lines_of_code[self.current_line][2]
            was_current_line = self.current_line
            self.lines_of_code[self.current_line][0](
                self.lines_of_code[self.current_line][1]
            )
            if self.current_line == was_current_line:
                self.current_line += 1
        return self.return_value

    def unpack_sequence_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        sequence = self.pop()
        for item in reversed(sequence):
            self.push(item)

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def load_build_class_op(self, arg: tp.Any) -> tp.Any:
        self.push(builtins.__build_class__)

    def build_string_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            ans = ""
            le = []
            for i in range(0, n):
                le.append(self.pop())
            for x in reversed(le):
                ans += x
            self.push(ans)
        else:
            self.push(str())

    def build_list_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            self.push(returned)
        else:
            self.push([])

    def list_append_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        tos = self.pop()
        tos1 = self.data_stack[-n]
        # tos1 = list(tos1)
        # if tos1 is range:
        #     tos1 = list(tos1)
        list.append(tos1, tos)

    def set_add_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        tos = self.pop()
        tos1 = self.data_stack[-n]
        # tos1 = list(tos1)
        # if tos1 is range:
        #     tos1 = list(tos1)
        set.add(tos1, tos)

    def dict_update_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        tos = self.pop()
        tos1 = self.data_stack[-n]
        # tos1 = list(tos1)
        # if tos1 is range:
        #     tos1 = list(tos1)
        dict.update(tos1, tos)

    def map_add_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        tos = self.pop()
        tos1 = self.pop()
        tos2 = self.data_stack[-n]
        # tos1 = list(tos1)
        # if tos1 is range:
        #     tos1 = list(tos1)
        dict.__setitem__(tos2, tos1, tos)

    def build_tuple_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            self.push(tuple(returned))
        else:
            self.push(tuple([]))

    def kw_names_op(self, n: tp.Any) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        # print(n, self.locals[n])
        # print(n, self.data_stack)
        assert self.current_arg is not None
        self.kw_names = self.code.co_consts[self.current_arg]   # type: ignore

        # self.push(self.locals[n])

    def delete_name_op(self, n: tp.Any) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        del self.locals[n]

    def delete_fast_op(self, n: tp.Any) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        del self.locals[n]

    def delete_global_op(self, n: tp.Any) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        del self.globals[n]

    def build_set_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            self.push(set(returned))
        else:
            self.push(set([]))

    def set_update_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        tos = self.pop()
        tos1 = self.pop()
        if tos1 is None:
            tos1 = set([])
        tos1.update(tos)
        self.push(tos1)

    def build_const_key_map_op(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        keys = self.pop()
        dict = {}
        values = []
        for key in keys:
            values.append(self.pop())
        for key, val in zip(keys, reversed(values)):
            dict[key] = val
        self.push(dict)

    def binary_subscr_op(self, arg: tp.Any) -> None:
        tos = self.pop()
        tos1 = self.pop()
        tos = tos1[tos]
        self.push(tos)

    def check_exc_match_op(self, arg: tp.Any) -> None:
        exc_type_to_check = self.pop()

        exc_instance_or_type = self.pop()

        is_match = False
        if isinstance(exc_instance_or_type, type):
            is_match = issubclass(exc_instance_or_type, exc_type_to_check)
        else:
            is_match = isinstance(exc_instance_or_type, exc_type_to_check)

        self.push(is_match)

    def delete_subscr_op(self, arg: int) -> None:
        tos = self.pop()
        tos1 = self.pop()
        del tos1[tos]

    def unary_negative_op(self, arg: int) -> None:
        tos = self.pop()
        self.push(-tos)

    def unary_positive_op(self, arg: int) -> None:
        tos = self.pop()
        self.push(+tos)

    def store_attr_op(self, arg: tp.Any) -> None:
        tos = self.pop()
        tos1 = self.pop()
        tos.arg = tos1
        self.push(tos)

    def load_attr_op(self, arg: tp.Any) -> None:
        tos = self.pop()
        self.push(getattr(tos, arg))

    def delete_attr_op(self, arg: tp.Any) -> None:
        tos = self.pop()
        del tos.arg

    def import_name_op(self, arg: tp.Any) -> None:
        tos = self.pop()
        tos1 = self.pop()
        module_name = arg

        # Import the module using the built-in `__import__` function.
        self.push(__import__(module_name, globals(), locals(), tos, tos1))

    def import_from_op(self, arg: tp.Any) -> None:
        tos = self.pop()
        attribute_name = arg

        attribute = getattr(tos, attribute_name)

        self.push(attribute)

    def load_method_op(self, arg: tp.Any) -> None:
        obj = self.pop()

        if hasattr(obj, arg):
            self.push(getattr(obj.__class__, arg))
            self.push(obj)
        else:
            self.push(None)
            self.push(getattr(obj, arg))

    def unary_not_op(self, arg: int) -> None:
        tos = self.pop()
        self.push(not tos)

    def build_map_op(self, arg: int) -> None:
        a = []
        for i in range(0, 2 * arg):
            a.append(self.pop())
        dict = {}
        for i in range(0, arg):
            dict[a[2 * arg - 1 - 2 * i]] = a[2 * arg - 2 - 2 * i]
        self.push(dict)

    def unary_invert_op(self, arg: int) -> None:
        tos = self.pop()
        self.push(~tos)

    def make_cell_op(self, arg: tp.Any) -> None:
        cell = ()
        if arg is not None:
            cell = (arg,)  # type: ignore
        self.push(cell)

    def get_len_op(self, arg: tp.Any) -> None:
        self.push(len(self.pop()))

    def list_extend_op(self, arg: int) -> None:
        # print(self.data_stack)
        tos = self.pop()
        tos1 = self.pop()
        # print(self.data_stack)
        tos1.extend(tos)
        self.push(tos1)
        # print(self.data_stack)

    def format_value_op(self, arg: tp.Any) -> None:
        val = tp.Any
        fmt_spec = tp.Any
        if arg[1]:
            fmt_spec = self.pop()
            val = self.pop()
        else:
            fmt_spec = ""
            val = self.pop()
        if arg[0] is str:
            val = str(val)
        elif arg[0] is repr:
            val = repr(val)
        elif arg[0] is ascii:
            val = ascii(val)
        self.push(format(val, fmt_spec))  # type: ignore

    def store_subscr_op(self, arg: int) -> None:
        # IDK what's wrong
        tos = self.pop()
        tos1 = self.pop()
        tos2 = self.pop()
        start, end = tos.start, tos.stop
        if start is None:
            start = 0
        if end is None:
            end = len(tos1)
        # print(tos1, start, end, tos2, " wtf")
        tos1[slice(start, end)] = tos2
        # self.push(tos2)
        self.push(tos1)
        # self.push(tos)

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """
        args = self.popn(arg)
        kwargs = {}  # type: ignore
        if self.kw_names:
            key_args = args[-len(self.kw_names):]   # type: ignore

            args = args[: -len(self.kw_names)]
            for i in range(0, len(key_args)):
                kwargs[self.kw_names[i]] = key_args[i]

        self.kw_names = ()  # type: ignore

        callable_obj = self.pop()
        if len(self.data_stack) > 0 and self.data_stack[-1] is not None:
            # [callable, self, positional args, named args]
            self_obj = self.pop()
            result = self_obj(callable_obj, *args, **kwargs)
        else:
            self.pop()
            # [NULL, callable, positional args, named args]
            result = callable_obj(*args, **kwargs)

        self.push(result)

    def build_slice_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """
        tos = self.pop()
        tos1 = self.pop()
        if arg == 2:
            self.push(slice(tos1, tos))
            # print("asd")
        else:
            # print("wtf")
            tos2 = self.pop()
            self.push(slice(tos2, tos1, tos))

    def pop_jump_forward_if_true_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_JUMP_FORWARD_IF_TRUE
        """
        ok = self.pop()
        if ok:
            self.update_position(arg)

    def pop_jump_backward_if_true_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_JUMP_FORWARD_IF_TRUE
        """
        ok = self.pop()
        if ok:
            self.update_position(arg)

    def pop_jump_backward_if_false_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_JUMP_FORWARD_IF_TRUE
        """
        ok = self.pop()
        if not ok:
            self.update_position(arg)

    def pop_jump_forward_if_none_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_JUMP_FORWARD_IF_TRUE
        """
        ok = self.pop()
        if ok is None:
            self.update_position(arg)

    def pop_jump_forward_if_false_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_JUMP_FORWARD_IF_TRUE
        """
        ok = self.pop()
        if not ok:
            self.update_position(arg)

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_NAME
        """
        # TODO: parse all scopes
        if arg == "print":
            self.push(print)
        elif arg == "range":
            self.push(range)
        elif arg == "str":
            self.push(str)
        elif arg == "list":
            self.push(list)
        elif arg == "object":
            self.push(object)
        elif arg == "isinstance":
            self.push(isinstance)
        elif arg == "float":
            self.push(float)
        elif arg == "sorted":
            self.push(sorted)
        elif arg == "len":
            self.push(len)
        elif arg == "hasattr":
            self.push(hasattr)
        elif arg == "__name__":
            self.push(__name__)
        elif arg == "exec":
            self.push(exec)
        elif arg in self.locals:
            self.push(self.locals[arg])
        else:
            self.push(None)

    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        code = self.pop()  # the code associated with the function (at TOS1)

        # TODO: use arg to parse function defaults
        defaults = ()
        kwdefaults = {}

        if arg & 2:
            kwdefaults = self.pop()

        if arg & 1:
            defaults = self.pop()

        # if defaults is None:
        #     defaults = ()

        # if kwdefaults is None:
        #     kwdefaults = {}

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            pos_only_arg_count = code.co_posonlyargcount
            kw_only_arg_count = code.co_kwonlyargcount
            positional_and_keyword_arg_names = list(
                code.co_varnames[: code.co_argcount]
            )
            positional_only_args = list(code.co_varnames[:pos_only_arg_count])
            keyword_only_args = list(
                code.co_varnames[
                    code.co_argcount: code.co_argcount + kw_only_arg_count
                ]
            )

            binds = {}

            has_varargs = code.co_flags & CO_VARARGS
            has_varkw = code.co_flags & CO_VARKEYWORDS

            pos_args_name = "args"
            kw_args_name = "kwargs"

            if has_varargs:
                pos_args_name = code.co_varnames[
                    code.co_argcount + kw_only_arg_count:
                ][0]

            if has_varkw:
                if has_varargs:
                    kw_args_name = code.co_varnames[
                        code.co_argcount + kw_only_arg_count + 1:
                    ][0]
                else:
                    kw_args_name = code.co_varnames[
                        code.co_argcount + kw_only_arg_count:
                    ][0]

            given_args = list(args)

            if has_varargs:
                binds[pos_args_name] = tuple(
                    given_args[len(positional_and_keyword_arg_names):]
                )
            # else:
            #     extra_pos_args = given_args[len(positional_and_keyword_arg_names) :]
            # if extra_pos_args:
            #     raise TypeError(ERR_TOO_MANY_POS_ARGS)

            if has_varkw:
                binds[kw_args_name] = {}  # type: ignore

            for name, value in zip(positional_and_keyword_arg_names, given_args):
                # if name in binds:
                #     raise TypeError(ERR_MULT_VALUES_FOR_ARG)
                binds[name] = value

            for name, value in kwargs.items():
                if name in positional_only_args:
                    if has_varkw:
                        # if name in binds[kw_args_name]:
                        #     TypeError(ERR_MULT_VALUES_FOR_ARG)
                        # else:
                        binds[kw_args_name][name] = value  # type: ignore
                    # else:
                    #     raise TypeError(ERR_POSONLY_PASSED_AS_KW)
                # if name in binds:
                #     raise TypeError(ERR_MULT_VALUES_FOR_ARG)
                if (
                    name not in keyword_only_args
                    and name not in positional_and_keyword_arg_names
                ):
                    if has_varkw:
                        binds[kw_args_name][name] = value  # type: ignore
                    # else:
                    #     raise TypeError(ERR_TOO_MANY_KW_ARGS)
                else:
                    binds[name] = value

            for name, value in zip(
                positional_and_keyword_arg_names[-len(defaults):], defaults
            ):
                binds.setdefault(name, value)
            for name, value in kwdefaults.items():
                binds.setdefault(name, value)

            # missing_args = [
            #     name for name in positional_and_keyword_arg_names if name not in binds
            # ]

            # print(binds)

            # if missing_args:
            #     raise TypeError(ERR_MISSING_POS_ARGS)

            # for name in keyword_only_args:
            #     if name not in binds:
            #         raise TypeError(ERR_MISSING_KWONLY_ARGS)

            parsed_args: dict[str, tp.Any] = binds
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)
            return frame.run()

        self.push(f)

    def update_position(self, arg: int) -> None:
        """
        sets the position to arg (offset)
        """
        self.current_line = self.offset_to_line[arg]
        # self.current_position += arg
        # i = 0
        # cur = 0
        # while cur < self.current_position and i < len(self.lines_of_code):
        #     if DEBUG:
        #         print("update_pos ", cur)
        #     cur += self.lengths_of_instr[i]
        #     i += 1
        # self.current_line = i - 1

    def nop_op(self, arg) -> None:  # type: ignore
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-NOP
        """
        pass

    def contains_op_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-NOP
        """
        a = self.pop()
        b = self.pop()
        if arg == 0:
            self.push(b in a)
        else:
            self.push(b not in a)

    def is_op_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-NOP
        """
        a = self.pop()
        b = self.pop()
        if arg == 0:
            self.push(b is a)
        else:
            self.push(b is not a)

    def jump_backward_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-JUMP_BACKWARD
        """
        self.update_position(arg)

    def jump_backward_no_interrupt_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-JUMP_BACKWARD
        """
        self.update_position(arg)

    def jump_forward_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-JUMP_BACKWARD
        """
        self.update_position(arg)

    def jump_if_true_or_pop_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-JUMP_BACKWARD
        """
        tos = self.pop()
        if tos:
            self.push(tos)
            self.update_position(arg)

    def jump_if_false_or_pop_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-JUMP_BACKWARD
        """
        tos = self.pop()
        if not tos:
            self.push(tos)
            self.update_position(arg)

    def get_iter_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-GET_ITER
        """
        arr = self.pop()
        self.push(iter(arr))

    def for_iter_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-FOR_ITER
        """
        iterator = self.pop()
        try:
            item = next(iterator)
            self.push(iterator)
            self.push(item)
        except StopIteration:
            self.update_position(arg)

    def binary_op_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-BINARY_OP
        """
        b = self.pop()
        a = self.pop()
        if arg == 13:
            if a is None and b is None:
                self.push(0)
                return
            self.push(a + b)
        elif arg == 6:
            self.push(a % b)
        elif arg == 0:
            if a is None:
                self.push(b)
                return
            self.push(a + b)
        elif arg == 21:
            self.push(a**b)
        elif arg == 18:
            self.push(a * b)
        elif arg == 15:
            self.push(a // b)
        elif arg == 19:
            self.push(a % b)
        elif arg == 13:
            self.push(a + b)
        elif arg == 23:
            self.push(a - b)
        elif arg == 16:
            self.push(a << b)
        elif arg == 22:
            self.push(a >> b)
        elif arg == 14:
            self.push(a & b)
        elif arg == 20:
            self.push(a | b)
        elif arg == 25:
            self.push(a ^ b)
        elif arg == 24:
            self.push(a / b)
        elif arg == 1:
            self.push(a & b)
        elif arg == 7:
            self.push(a | b)
        elif arg == 12:
            self.push(a ^ b)
        elif arg == 3:
            self.push(a << b)
        elif arg == 9:
            self.push(a >> b)
        elif arg == 10:
            self.push(a - b)
        elif arg == 2:
            self.push(a // b)
        elif arg == 4:
            self.push(a @ b)
        elif arg == 5:
            self.push(a * b)
        elif arg == 8:
            self.push(a**b)
        elif arg == 11:
            self.push(a / b)
        elif arg == 17:
            self.push(a @ b)

    def raise_varargs_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-BINARY_OP
        """
        if arg == 0:
            raise
        elif arg == 1:
            a = self.pop()
            raise a
        elif arg == 2:
            a = self.pop()
            b = self.pop()
            raise b(a)

    def reraise_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-BINARY_OP
        """
        ex = self.pop()
        if arg is not None:
            what = self.pop
            self.f_lasti = what  # type: ignore
        raise ex

    def push_exc_info_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-BINARY_OP
        """
        tos = self.pop()

        self.push(exc_info())

        self.push(tos)

    def pop_except_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-BINARY_OP
        """
        self.pop()

    def load_assertion_error_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-BINARY_OP
        """
        self.push(AssertionError)

    def compare_op_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-BINARY_OP
        """
        b = self.pop()
        a = self.pop()
        # print(a, b, arg)
        if arg == "<":
            self.push(a < b)
        elif arg == ">":
            self.push(a > b)
        elif arg == "<=":
            self.push(a <= b)
        elif arg == ">=":
            self.push(a >= b)
        elif arg == "==":
            self.push(a == b)
        elif arg == "!=":
            self.push(a != b)

    def load_global_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        # TODO: parse all scopes
        if arg in self.builtins:
            self.push(self.builtins[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        else:
            self.push(None)

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def load_fast_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(self.locals[arg])

    def load_closure_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(self.locals[arg])

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()
        self.is_end = True

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()

        self.locals[arg] = const

    def store_fast_op(self, arg: tp.Any) -> None:
        # wrong
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[arg] = const

    def store_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_GLOBAL
        """
        const = self.pop()
        self.globals[arg] = const

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(
            code_obj,
            builtins.globals()["__builtins__"],
            globals_context,
            globals_context,
        )
        return frame.run()
