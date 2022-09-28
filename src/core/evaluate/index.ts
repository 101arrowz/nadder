import { broadcast, ndvInternals } from '../../util';
import * as ufuncOps from '../../util/ufunc/ops';
import * as linalgOps from '../../util/linalg';
import * as helpers from '../../util/helpers';
import { NDView } from '../ndarray';
import { DataType, dataTypeNames } from '../datatype';

const baseCtx = {
  ...ufuncOps,
  ...linalgOps,
  ...helpers,
  reshape: (arr: NDView, ...args: [number[]]) => {
    if (!arr || !arr[ndvInternals]) {
      throw new TypeError('cannot reshape non-ndarray')
    }
    if (args.length > 1) {
      throw new TypeError('must use array for dimensions in free function reshape');
    }
    return arr.reshape(args[0]);
  }
};

for (const k in dataTypeNames) {
  baseCtx[dataTypeNames[k]] = +k;
}

declare const enum TokenType {
  Operator = 0,
  Identifier = 1,
  Bracket = 2,
  Value = 3,
  Separator = 4,
  Keyword = 5
}

type Token = { t: Exclude<TokenType, TokenType.Value>; v: string } | {
  t: TokenType.Value;
  v: unknown;
};

const tokenTypes: Record<TokenType, string> = [
  'operator',
  'identifier',
  'bracket',
  'value',
  'separator',
  'keyword'
];

const keywords = new Set([
  'for',
  'while',
  'if',
  'else'
]);

// inplace ops
const opMap = {
  '+=': 'add',
  '-=': 'sub',
  '*=': 'mul',
  '/=': 'div',
  '%=': 'mod',
  '@=': 'matmul',
  '**=': 'pow',
  '<<=': 'shl',
  '>>=': 'shr',
  '&=': 'bitand',
  '^=': 'bitxor',
  '|=': 'bitor',
  '&&=': 'and',
  '||=': 'or',
};

const unaryOpMap = {
  '+': 'pos',
  '-': 'neg',
  '!': 'not',
  '~': 'bitnot'
};

const symbolic = Symbol();

type Context = {
  // environment
  e: Record<string, unknown>;
  // inputs
  i: Record<string, unknown>;
};

type ASTNode = (ctx: Context) => unknown;

type FilteredArgs<T extends unknown[]> = {
  [K in keyof T]: T[K] extends Argument<infer U> ? U : never;
}[number];

/**
 * A symbolic argument in a DSL function
 */
export interface Argument<T extends string | number | symbol> {
  [symbolic]: T;
}

/**
 * Creates an argument to a function written in the DSL
 * @param name The name of the argument
 * @returns A symbolic argument that can be used in `parse`
 */
export function argument<T extends string | number | symbol>(name: T): Argument<T> {
  if (typeof name != 'string' && typeof name != 'number' && typeof name != 'symbol') {
    throw new TypeError('argument name must be a string, number, or symbol');
  }
  return {
    [symbolic]: name
  };
};

export type ParsedFunction<T extends unknown[]> =
  FilteredArgs<T> extends never
    ? () => unknown
    : (args: Record<FilteredArgs<T>, unknown>) => unknown;

/**
 * Parses an expression of operations potentially applied to ndarrays. This allows for more
 * natural syntax, e.g. using the `*` operator instead of `mul()`
 * @param code The code snippets to evaluate
 * @param args The arguments to interpolate between the code snippets
 * @returns The result of evaluating the provided expression
 * @example
 * ```js
 * import { parse, argument, arange } from 'nadder';
 * 
 * const b = arange(5 * 5).reshape([5, 5])['::-1'];
 * 
 * // Typically called as a tagged template
 * const biasedMul = parse`
 *   ${argument('a')} * ${b} + [5, 4, 3, 2, 1]
 * `;
 * 
 * const result = biasedMul({
 *   a: arange(5 * 5).reshape([5, 5])
 * });
 * ```
 */
export function parse<T extends unknown[]>(code: readonly string[], ...args: T): ParsedFunction<T> {
  if (!code.length || args.length != code.length - 1) {
    throw new TypeError('invalid arguments to evaluate');
  }
  let curStringInd = 0;
  let curInput = code[0];
  let tokens: Token[] = [];
  o: while (true) {
    curInput = curInput.trimStart();
    while (!curInput) {
      if (++curStringInd >= code.length) break o;
      tokens.push({ t: TokenType.Value, v: args[curStringInd - 1] });
      curInput = code[curStringInd].trimStart();
    }
    let char = curInput[0];
    switch (char) {
      case ',':
      case ':':
      case ';':
        curInput = curInput.slice(1);
        tokens.push({ t: TokenType.Separator, v: char });
        break;
      case '~':
        curInput = curInput.slice(1);
        tokens.push({ t: TokenType.Operator, v: char });
        break;
      case '*':
        if (curInput[1] == '*') char += '*';
        // fallthrough
      case '+':
      case '-':
      case '%':
      case '@':
      case '^':
      case '=':
      case '!':
        if (curInput[1] == '=') char += '=';
        curInput = curInput.slice(char.length);
        tokens.push({ t: TokenType.Operator, v: char });
        break;
      case '>':
      case '<':
      case '&':
      case '|':
      case '/': {
        let ind = 1;
        if (curInput[ind] == char) {
          char += char;
          ++ind;
        }
        if (curInput[ind] == '=') char += '='
        curInput = curInput.slice(char.length);
        tokens.push({ t: TokenType.Operator, v: char });
        break;
      }
      case '(':
      case ')':
      case '[':
      case ']':
      case '{':
      case '}':
        tokens.push({ t: TokenType.Bracket, v: char });
        curInput = curInput.slice(1);
        break;
      case '.': {
        if (curInput[1] == '.' && curInput[2] == '.') {
          tokens.push({ t: TokenType.Operator, v: '...' });
          curInput = curInput.slice(3);
          break;
        }
        // inverted to support NaN
        if (!(curInput[1] >= '0' && curInput[1] <= '9')) {
          tokens.push({ t: TokenType.Operator, v: char });
          curInput = curInput.slice(1);
          break;
        }
      }
      default: {
        const code = char.charCodeAt(0);
        // 0-9, or point
        if (code == 46 || (code > 47 && code < 58)) {
          let num = char;
          let seenPoint = code == 46;
          let ind = 1;
          while (ind < curInput.length) {
            const newCode = curInput.charCodeAt(ind);
            // 0-9
            if (newCode < 48 || newCode > 57) {
              if (newCode != 46 || seenPoint) break;
              seenPoint = true;
            };
            num += curInput[ind++];
          }
          if (num.length > 1 && code == 48) {
            throw new TypeError('leading zeros are not allowed in integer literals');
          }
          tokens.push({ t: TokenType.Value, v: +num });
          curInput = curInput.slice(ind);
          break;
        }
        // a-zA-Z
        if (code > 96 && code < 123 || code > 64 && code < 91) {
          let ident = char;
          let ind = 1;
          while (ind < curInput.length) {
            const newCode = curInput.charCodeAt(ind);
            // a-zA-Z0-9_
            if ((newCode < 97 || newCode > 122) &&
              (newCode < 65 || newCode > 90) &&
              (newCode < 48 || newCode > 57) &&
              (newCode != 95)
            ) break;
            ident += curInput[ind++];
          }
          tokens.push({ t: keywords.has(ident) ? TokenType.Keyword : TokenType.Identifier, v: ident });
          curInput = curInput.slice(ind);
          break;
        }
        throw new SyntaxError(`could not parse token from ${char}`);
      }
    }
  }

  // popping is faster than shifting
  tokens.reverse();

  const cur = () => tokens[tokens.length - 1];

  const expect = (fn?: (v: Token) => unknown) => {
    if (cur()) {
      if (!fn || fn(cur())) return tokens[tokens.length - 1];
      throw new TypeError(`unexpected ${tokenTypes[cur().t]} ${cur().v}`);
    }
    throw new TypeError('unexpected end of input');
  }

  const check = (fn: (v: Token) => unknown) => {
    if (cur() && fn(cur())) return cur();
  }

  const tryCall = (fn: ASTNode, args: ASTNode[]) => (ctx: Context) => {
    const callable = fn(ctx);
    const params = args.map(arg => arg(ctx));
    if (typeof callable != 'function') {
      throw new TypeError(`attempted to call non-function ${callable}`);
    }
    return callable(...params);
  }

  const slice = () => {
    let sliceParts: ASTNode[] = [];
    while (expect().t != TokenType.Bracket || cur().v != ']') {
      // hack for np.newaxis (+)
      while (
        (expect().t == TokenType.Separator && cur().v != ';') ||
        (cur().t == TokenType.Operator && (cur().v == '+' || cur().v == '...'))
      ) {
        const elem = tokens.pop();
        sliceParts.push(() => elem.v);
      }
      if (expect().t == TokenType.Bracket && cur().v == ']') break;
      sliceParts.push(expr());
    }
    tokens.pop();
    return (ctx: Context) => sliceParts.map(v => v(ctx)).join('');
  };

  const funcArgs = () => {
    const token = expect();
    if (token.t == TokenType.Bracket && token.v == ')') {
      tokens.pop();
      return [];
    }
    let postArg: Record<string, ASTNode>;
    const args: ASTNode[] = [];
    while (true) {
      let pos = 1;
      if (check(v => v.t == TokenType.Identifier)) {
        const tok = tokens.pop();
        if (check(v => v.t == TokenType.Operator && v.v == '=')) {
          tokens.pop();
          if (!postArg) postArg = {};
          postArg[tok.v as string] = expr();
          pos = 0;
        } else tokens.push(tok);
      }
      if (pos) {
        if (postArg) {
          throw new SyntaxError('cannot have keyword arguments after positional arguments');
        }
        args.push(expr());
      }
      const sep = expect();
      tokens.pop();
      if (sep.t == TokenType.Bracket && sep.v == ')') break;
      if (sep.t != TokenType.Separator || sep.v != ',') {
        throw new SyntaxError(`expected comma in argument list, got ${tokenTypes[sep.t]} ${sep.v}`);
      }
    }
    if (postArg) {
      args.push(ctx => {
        const result: Record<string, unknown> = {};
        for (const k in postArg) {
          result[k] = postArg[k](ctx);
        }
        return result;
      })
    }
    return args;
  };

  const condVal = (ret: unknown) => {
    if (ret && ret[ndvInternals]) {
      if (!(ret as NDView).ndim) return (ret as NDView).get();
      throw new TypeError('cannot check truthiness of array; use any() or all()');
    }
    return ret;
  }

  const cond = (val: ASTNode): ASTNode => (ctx: Context) => condVal(val(ctx));

  const maybeBracket = (val: ASTNode): ASTNode => {
    if (check(bracket =>
      (bracket.t == TokenType.Bracket && (bracket.v == '(' || bracket.v == '[')) ||
      (bracket.t == TokenType.Operator && (bracket.v == '.'))
    )) {
      const bracket = tokens.pop() as { t: TokenType.Identifier | TokenType.Operator; v: string };
      if (bracket.v == '(') {
        return maybeBracket(tryCall(val, funcArgs()));
      } else if (bracket.v == '[') {
        const res = slice();
        return maybeBracket(ctx => val(ctx)[res(ctx)]);
      } else {
        const ident = expect(t => t.t == TokenType.Identifier).v as string;
        tokens.pop();
        return maybeBracket(ctx => {
          const v = val(ctx);
          const result = v[ident];
          return typeof result == 'function'
            ? result.bind(v)
            : result;
        })
      }
    }
    return val;
  };

  const unit = (): ASTNode => {
    const token = expect();
    if (token.t == TokenType.Value) {
      tokens.pop();
      let op = token.v && token.v[symbolic] != null
        ? (ctx: Context) => ctx.i[token.v[symbolic]]
        : () => token.v;
      return maybeBracket(op);
    }
    if (token.t == TokenType.Operator) {
      tokens.pop();
      const operand = unit();
      const op = unaryOpMap[token.v];
      if (!op) throw new SyntaxError(`could not parse unary operator ${token.v}`);
      return maybeBracket(ctx => baseCtx[op](operand(ctx) as number));
    }
    if (token.t == TokenType.Identifier) {
      const name = token.v;
      tokens.pop();
      return maybeBracket(ctx => {
        if (name in ctx.e) return ctx.e[name];
        throw new ReferenceError(`unknown identifier ${name}`);
      });
    }
    if (token.t == TokenType.Bracket) {
      let result: ASTNode;
      if (token.v == '(') {
        tokens.pop();
        result = expr();
        if (expect().t != TokenType.Bracket || cur().v != ')') {
          throw new SyntaxError('expected closing bracket');
        }
        tokens.pop();
      } else if (token.v == '[') {
        tokens.pop();
        if (expect().t == TokenType.Bracket && cur().v == ']') {
          tokens.pop();
          result = () => [];
        } else {
          let vals: ASTNode[] = [expr()];
          while (expect().t != TokenType.Bracket || cur().v != ']') {
            const sep = tokens.pop();
            if (sep.t != TokenType.Separator || sep.v != ',') {
              throw new SyntaxError(`expected comma in array literal, got ${tokenTypes[sep.t]} ${sep.v}`);
            }
            vals.push(expr());
          }
          tokens.pop();
          result = ctx => vals.map(val => val(ctx));
        }
      }
      return maybeBracket(result);
    }
    throw new SyntaxError('could not parse expression');
  };

  const make2NumOp = (op: string, left: ASTNode, makeRight: () => ASTNode): ASTNode => {
    tokens.pop();
    const right = makeRight();
    return (ctx: Context) => {
      const l = left(ctx) as number;
      const r = right(ctx) as number;
      return baseCtx[op](l, r);
    }
  }

  const powExpr = () => {
    let base = unit();
    if (check(t => t.t == TokenType.Operator && t.v == '**')) {
      tokens.pop();
      const right = powExpr();
      return (ctx: Context) => {
        const l = base(ctx) as number;
        const r = right(ctx) as number;
        if (l === Math.E) return baseCtx.exp(r);
        if (l === 2) return baseCtx.exp2(r);
        return baseCtx.pow(l, r);
      };
    }
    return base;
  };

  const makeLeftAssoc = (ops: Record<string, string>, lower: () => ASTNode) => () => {
    let left = lower();
    while (check(t => t.t == TokenType.Operator)) {
      const op = ops[cur().v as string];
      if (!op) break;
      left = make2NumOp(op, left, lower);
    }
    return left;
  }

  const mulExpr = makeLeftAssoc({
    '*': 'mul',
    '/': 'div',
    '//': 'fdiv',
    '%': 'mod',
    '@': 'matmul'
  }, powExpr);

  const addExpr = makeLeftAssoc({
    '+': 'add',
    '-': 'sub'
  }, mulExpr);

  const shiftExpr = makeLeftAssoc({
    '<<': 'shl',
    '>>': 'shr'
  }, addExpr);

  const makeBitop = (op: string, name: string, lower: () => ASTNode) => () => {
    let left = lower();
    while (check(t => t.t == TokenType.Operator && t.v == op)) {
      const oldleft = left;
      tokens.pop();
      const right = lower();
      left = (ctx: Context) => {
        const lhs = oldleft(ctx);
        const rhs = right(ctx);
        let opn = name;
        if ((broadcast(lhs)[0]['t'].t & broadcast(rhs)[0]['t'].t) != DataType.Bool) opn = 'bit' + name;
        return baseCtx[opn](lhs, rhs);
      };
    }
    return left;
  };

  const bandExpr = makeBitop('&', 'and', shiftExpr);
  const bxorExpr = makeBitop('^', 'xor', bandExpr);
  const borExpr = makeBitop('|', 'or', bxorExpr);

  const relExpr = makeLeftAssoc({
    '<': 'lt',
    '<=': 'lte',
    '>': 'gt',
    '>=': 'gte'
  }, borExpr);

  const cmpExpr = makeLeftAssoc({
    '==': 'eq',
    '!=': 'ne'
  }, relExpr);

  const boolExpr = () => {
    let left = cmpExpr();
    while (check(t => t.t == TokenType.Operator)) {
      const oldleft = cond(left);
      if (cur().v == '&&') {
        tokens.pop();
        const right = cmpExpr();
        left = (ctx: Context) => {
          const lhs = oldleft(ctx);
          if (!lhs) return lhs;
          return right(ctx);
        };
      } else if (cur().v == '||') {
        tokens.pop();
        const right = cmpExpr();
        left = (ctx: Context) => {
          const lhs = oldleft(ctx);
          if (lhs) return lhs;
          return right(ctx);
        };
      } else break;
    }
    return left;
  }

  const expr = () => {
    // TODO: more stuff?
    return boolExpr();
  };

  const stmt = () => {
    let result: ASTNode = () => {};
    while (true) {
      const oldresult = result;
      if (!cur() || cur().t == TokenType.Bracket && cur().v == '}') break;
      if (cur().t == TokenType.Separator && cur().v == ';') {
        tokens.pop();
        continue;
      }
      if (check(t => t.t == TokenType.Keyword && t.v == 'for')) {
        tokens.pop();
        const name = expect(t => t.t == TokenType.Identifier).v as string;
        tokens.pop();
        expect(t => t.t == TokenType.Identifier && t.v == 'in');
        tokens.pop();
        const toIter = expr();
        expect(t => t.t == TokenType.Bracket && t.v == '{');
        tokens.pop();
        const body = stmt();
        expect(t => t.t == TokenType.Bracket && t.v == '}');
        tokens.pop();
        result = (ctx: Context) => {
          oldresult(ctx);
          const it = toIter(ctx);
          if (!it || typeof it[Symbol.iterator] != 'function') {
            throw new TypeError('cannot iterate over non-iterable');
          }
          for (const val of it as unknown[]) {
            ctx.e[name] = val;
            body(ctx);
          }
        };
        continue;
      } else if (check(t => t.t == TokenType.Keyword && t.v == 'while')) {
        tokens.pop();
        const condition = cond(expr());
        expect(t => t.t == TokenType.Bracket && t.v == '{');
        tokens.pop();
        const body = stmt();
        expect(t => t.t == TokenType.Bracket && t.v == '}');
        tokens.pop();
        result = (ctx: Context) => {
          oldresult(ctx);
          while (condition(ctx)) {
            body(ctx);
          }
        };
        continue;
      } else if (check(t => t.t == TokenType.Keyword && t.v == 'if')) {
        const parseIf = () => {
          tokens.pop();
          const condition = cond(expr());
          expect(t => t.t == TokenType.Bracket && t.v == '{');
          tokens.pop();
          const body = stmt();
          expect(t => t.t == TokenType.Bracket && t.v == '}');
          tokens.pop();
          let elseBody: ASTNode = () => {};
          if (check(t => t.t == TokenType.Keyword && t.v == 'else')) {
            tokens.pop();
            if (check(t => t.t == TokenType.Keyword && t.v == 'if')) {
              elseBody = parseIf();
            } else {
              expect(t => t.t == TokenType.Bracket && t.v == '{');
              tokens.pop();
              elseBody = stmt();
              expect(t => t.t == TokenType.Bracket && t.v == '}');
              tokens.pop();
            }
          }
          return (ctx: Context) => {
            if (condition(ctx)) {
              body(ctx);
            } else {
              elseBody(ctx);
            }
          };
        }
        const ext = parseIf();
        result = (ctx: Context) => {
          oldresult(ctx);
          ext(ctx);
        }
        continue;
      } else if (check(t => t.t == TokenType.Identifier || t.t == TokenType.Value)) {
        let prevTokens = tokens.slice();
        const token = tokens.pop();
        const handleMod = (ctx: Context, op: string, a: unknown, mb: (ctx: Context) => unknown, run: (val: unknown) => unknown) => {
          if (op == 'and') {
            if (condVal(a)) {
              let val = mb(ctx);
              if (a && a[ndvInternals]) (a as NDView).set(val);
              else run(val);
            }
          } else if (op == 'or') {
            if (!condVal(a)) {
              let val = mb(ctx);
              if (a && a[ndvInternals]) (a as NDView).set(val);
              else run(val);
            }
          } else {
            const b = mb(ctx);
            if (
              (op == 'bitand' || op == 'bitor' || op == 'bitxor') &&
              (broadcast(a)[0]['t'].t & broadcast(b)[0]['t'].t) == DataType.Bool
            ) op = op.slice(3);
            if (a && a[ndvInternals]) {
              baseCtx[op](a, b, { out: a });
            } else {
              run(baseCtx[op](a, b));
            }
          } 
        }
        let tgt = {
          g: (ctx: Context) => {
            if (token.t == TokenType.Identifier) return ctx.e[token.v];
            else if (token.v && token.v[symbolic] != null) return ctx.i[token.v[symbolic]];
            else return token.v;
          },
          s: (v: ASTNode) => {
            if (token.t == TokenType.Value && (!token.v || token.v[symbolic] == null)) {
              throw new SyntaxError('cannot set non-symbolic value');
            }
            return (ctx: Context) => {
              if (token.t == TokenType.Identifier) ctx.e[token.v] = v(ctx);
              else ctx.i[token.v[symbolic]] = v(ctx);
            }
          },
          m: (op: string, b: ASTNode) => {
            if (token.t == TokenType.Value &&
              (!token.v || (token.v[symbolic] == null && !token.v[ndvInternals]))
            ) {
              throw new SyntaxError('cannot modify non-symbolic, non-ndarray value');
            }
            return (ctx: Context) => {
              if (token.t == TokenType.Identifier) {
                handleMod(ctx, op, ctx.e[token.v], b, val => {
                  ctx.e[token.v] = val;
                });
              } else if (token.v[symbolic] != null) {
                handleMod(ctx, op, ctx.i[token.v[symbolic]], b, val => {
                  ctx.i[token.v[symbolic]] = val;
                });
              } else {
                handleMod(ctx, op, token.v, b, () => {
                  throw new Error('unreachable');
                });
              }
            }
          }
        };
        while (check(bracket =>
          (bracket.t == TokenType.Bracket && (bracket.v == '(' || bracket.v == '[')) ||
          (bracket.t == TokenType.Operator && (bracket.v == '.'))
        )) {
          const bracket = tokens.pop();
          if (bracket.v == '(') {
            tgt = {
              g: tryCall(tgt.g, funcArgs()),
              s: () => {
                throw new TypeError('cannot assign to function call');
              },
              m: () => {
                throw new TypeError('cannot modify function call');
              }
            };
          } else if (bracket.v == '[') {
            const makeSlice = slice();
            const oldt = tgt;
            tgt = {
              g: (ctx: Context) => oldt.g(ctx)[makeSlice(ctx)],
              s: (v: ASTNode) => (ctx: Context) => {
                oldt.g(ctx)[makeSlice(ctx)] = v(ctx);
              },
              m: (op: string, b: ASTNode) => (ctx: Context) => {
                const base = oldt.g(ctx);
                const slice = makeSlice(ctx);
                const tgt = base[slice];
                handleMod(ctx, op, tgt, b, val => {
                  base[slice] = val;
                });
              }
            };
          } else {
            const ident = expect(t => t.t == TokenType.Identifier).v as string;
            tokens.pop();
            const oldt = tgt;
            tgt = {
              g: (ctx: Context) => {
                const v = oldt.g(ctx);
                const result = v[ident];
                return typeof result == 'function'
                  ? result.bind(v)
                  : result;
              },
              s: (v: ASTNode) => (ctx: Context) => {
                oldt.g(ctx)[ident] = v(ctx);
              },
              m: (op: string, b: ASTNode) => (ctx: Context) => {
                const base = oldt.g(ctx);
                const tgt = base[ident];
                handleMod(ctx, op, tgt, b, val => {
                  base[ident] = val;
                });
              }
            };
          }
        }
        if (check(t => t.t == TokenType.Operator)) {
          const op = tokens.pop().v as string;
          if (op == '=') {
            const set = tgt.s(expr());
            result = (ctx: Context) => {
              oldresult(ctx);
              set(ctx);
            };
          } else if (opMap[op]) {
            const modify = tgt.m(opMap[op], expr());
            result = (ctx: Context) => {
              oldresult(ctx);
              modify(ctx);
            };
          }
        }
        if (result == oldresult) {
          tokens = prevTokens;
          const next = expr();
          let ret = !check(t => t.t == TokenType.Separator && t.v == ';');
          result = (ctx: Context) => {
            oldresult(ctx);
            const val = next(ctx);
            if (ret) return val;
          }
          if (ret) break;
        }
      } else {
        const next = expr();
        let ret = !check(t => t.t == TokenType.Separator && t.v == ';');
        result = (ctx: Context) => {
          oldresult(ctx);
          const val = next(ctx);
          if (ret) return val;
        }
        if (ret) break;
      }
      expect(t => t.t == TokenType.Separator && t.v == ';');
    }
    return result;
  };

  // TODO: variables and loops
  
  const result = stmt();
  if (cur()) {
    throw new SyntaxError(`unexpected ${tokenTypes[cur().t]} ${cur().v}`);
  }

  return (args?: Record<FilteredArgs<T>, unknown>) => result({
    i: args || {},
    e: { ...baseCtx }
  });
}

/**
 * Evaluates an expression of operations potentially applied to ndarrays. This allows for more
 * natural syntax, e.g. using the `*` operator instead of `mul()`
 * @param code The code snippets to evaluate
 * @param args The values to interpolate between the code snippets
 * @returns The result of evaluating the provided expression
 * @example
 * ```js
 * import { evaluate, arange } from 'nadder';
 * 
 * // Typically called as a tagged template
 * const result = evaluate`
 *   ${arange(20).reshape(5, 4)} @ ${arange(30, 34)} +
 *   [5, 4, 3, 2, 1]
 * `;
 * ```
 */
export function evaluate(code: readonly string[], ...args: unknown[]): unknown {
  for (const arg of args) {
    if (arg && arg[symbolic] != null) {
      throw new TypeError('cannot evaluate expression with arguments');
    }
  }
  return parse(code, ...args)();
}
