import { ndvInternals } from '../../util';
import * as ufuncOps from '../../util/ufunc/ops';
import * as linalgOps from '../../util/linalg';
import * as helpers from '../../util/helpers';
import { NDView } from '../ndarray';

const ops = {
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
}

declare const enum TokenType {
  Operator = 0,
  Identifier = 1,
  Bracket = 2,
  Value = 3,
  Separator = 4
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
  'separator'
];

const symbolic = Symbol();

type Context = {
  // environment
  e: Record<string, unknown>;
  // inputs
  i: Record<string, unknown>;
};

type ASTNode = (ctx: Context) => unknown;

/**
 * Parses an expression of operations potentially applied to ndarrays. This allows for more
 * natural syntax, e.g. using the `*` operator instead of `mul()`
 * @param code The code snippets to evaluate
 * @param args The arguments to interpolate between the code snippets
 * @returns The result of evaluating the provided expression
 * @example
 * ```js
 * import { parse, arange } from 'nadder';
 * 
 * // Typically called as a tagged template
 * const biasedMul = parse`
 *   ${'a'} * ${'b'} + [5, 4, 3, 2, 1]
 * `;
 * 
 * const result = biasedMul({
 *   a: arange(5 * 5).reshape([5, 5]),
 *   b: arange(5 * 5).reshape([5, 5])['::-1']
 * });
 * ```
 */
export function parse<T extends string | symbol | number>(code: readonly string[], ...args: T[]) {
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
      tokens.push({ t: TokenType.Value, v: { [symbolic]: args[curStringInd - 1] } });
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
      case '*':
        if (curInput[1] == '*') char += '*';
        // fallthrough
      case '+':
      case '-':
      case '/':
      case '%':
      case '@':
      case '=':
      case '>':
      case '<':
        if (curInput[1] == '=') char += '=';
        curInput = curInput.slice(char.length);
        tokens.push({ t: TokenType.Operator, v: char });
        break;
      case '(':
      case ')':
      case '[':
      case ']':
      case '{':
      case '}':
        tokens.push({ t: TokenType.Bracket, v: char });
        curInput = curInput.slice(1);
        break;
      case '.':
        if (curInput[1] == '.' && curInput[2] == '.') {
          tokens.push({ t: TokenType.Operator, v: '...' });
          curInput = curInput.slice(3);
          break;
        }
      default:
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
          tokens.push({ t: TokenType.Identifier, v: ident });
          curInput = curInput.slice(ind);
          break;
        }
        throw new SyntaxError(`could not parse token from ${char}`);
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

  const maybeBracket = (val: ASTNode): ASTNode => {
    const bracket = cur();
    if (bracket && bracket.t == TokenType.Bracket && (bracket.v == '(' || bracket.v == '[')) {
      tokens.pop();
      if (bracket.v == '(') {
        // function call
        const token = expect();
        if (token.t == TokenType.Bracket && token.v == ')') {
          tokens.pop();
          return maybeBracket(tryCall(val, []));
        }
        const args: ASTNode[] = [expr()];
        while (expect().t != TokenType.Bracket || cur().v != ')') {
          const sep = tokens.pop();
          if (sep.t != TokenType.Separator || sep.v != ',') {
            throw new SyntaxError(`expected comma in argument list, got ${tokenTypes[sep.t]} ${sep.v}`);
          }
          args.push(expr());
        }
        tokens.pop();
        return maybeBracket(tryCall(val, args));
      } else if (bracket.v == '[') {
        // slice index
        let sliceParts: ASTNode[] = [];
        while (expect().t != TokenType.Bracket || cur().v != ']') {
          // hack for np.newaxis (+)
          while (
            (expect().t == TokenType.Separator && cur().v != ';') ||
            (cur().t == TokenType.Operator && (cur().v == '+' || cur().v == '...'))
          ) {
            sliceParts.push(() => tokens.pop().v);
          }
          if (expect().t == TokenType.Bracket && cur().v == ']') break;
          sliceParts.push(expr());
        }
        tokens.pop();
        return maybeBracket(ctx => val(ctx)[sliceParts.map(part => part(ctx)).join('')]);
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
      const op = token.v == '-' ? 'neg' : token.v == '+' ? 'pos' : null;
      if (!op) throw new SyntaxError(`could not parse unary operator ${token.v}`);
      return maybeBracket(ctx => ops[op](operand(ctx) as number));
    }
    if (token.t == TokenType.Identifier) {
      const name = token.v;
      tokens.pop();
      return maybeBracket(ctx => {
        if (name in ctx.i) return ctx.i[name];
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
      return ops[op](l, r);
    }
  }

  const powExpr = () => {
    let left = unit();
    while (check(t => t.t == TokenType.Operator)) {
      switch (cur().v) {
        case '**':
          tokens.pop();
          const right = unit();
          left = (ctx: Context) => {
            const l = left(ctx) as number;
            const r = right(ctx) as number;
            if (l === Math.E) return ops.exp(r);
            if (l === 2) return ops.exp2(r);
            return ops.pow(l, r);
          };
          break;
        default: return left;
      }
    }
    return left;
  };

  const mulExpr = () => {
    let left = powExpr();
    while (check(t => t.t == TokenType.Operator)) {
      switch (cur().v) {
        case '*':
          left = make2NumOp('mul', left, powExpr);
          break;
        case '/':
          left = make2NumOp('div', left, powExpr);
          break;
        case '%':
          left = make2NumOp('mod', left, powExpr);
          break;
        case '@':
          left = make2NumOp('matmul', left, powExpr);
          break;
        default: return left;
      }
    }
    return left;
  };

  const addExpr = () => {
    let left = mulExpr();
    while (check(t => t.t == TokenType.Operator)) {
      switch (cur().v) {
        case '+':
          left = make2NumOp('add', left, mulExpr);
          break;
        case '-':
          left = make2NumOp('sub', left, mulExpr);
          break;
        default: return left;
      }
    }
    return left;
  };

  const relExpr = () => {
    // TODO: relational operators
    return addExpr();
  };

  const expr = () => {
    // TODO: more stuff?
    return relExpr();
  };

  const stmt = () => {
    let result: ASTNode = () => {};
    while (true) {
      if (cur() && cur().t == TokenType.Bracket && cur().v == '}') break;
      if (cur() && cur().t == TokenType.Separator && cur().v == ';') {
        tokens.pop();
        continue;
      }
      if (check(t => t.t == TokenType.Identifier && t.v == 'for')) {
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
          const it = toIter(ctx);
          if (!it || typeof it[Symbol.iterator] != 'function') {
            throw new TypeError('cannot iterate over non-iterable');
          }
          for (const val of it as unknown[]) {
            ctx.e[name] = val;
            body(ctx);
          }
        };
      } else if (check(t => t.t == TokenType.Identifier && t.v == 'while')) {
        tokens.pop();
        const cond = expr();
        tokens.pop();
        expect(t => t.t == TokenType.Bracket && t.v == '{');
        tokens.pop();
        const body = stmt();
        expect(t => t.t == TokenType.Bracket && t.v == '}');
        tokens.pop();
        result = (ctx: Context) => {
          while (cond(ctx)) {
            body(ctx);
          }
        };
      } else if (check(t => t.t == TokenType.Identifier || (t.t == TokenType.Value && t.v && t.v[symbolic] != null))) {
        let prevTokens = tokens.slice();
        const token = tokens.pop();
        let tgt = {
          g: (ctx: Context) => {
            if (token.t == TokenType.Identifier) return ctx.e[token.v];
            else return ctx.i[token.v[symbolic]];
          },
          s: (v: ASTNode) => (ctx: Context) => {
            if (token.t == TokenType.Identifier) ctx.e[token.v] = v(ctx);
            else ctx.i[token.v[symbolic]] = v(ctx);
          },
          m: (op: string, b: ASTNode) => (ctx: Context) => {
            if (token.t == TokenType.Identifier) {
              let a = ctx.e[token.v];
              if (a && a[ndvInternals]) {
                ops[op](a, b(ctx), { out: a });
              } else {
                ctx.e[token.v] = ops[op](a, b(ctx));
              }
            }
            else {
              let a = ctx.i[token.v[symbolic]];
              if (a && a[ndvInternals]) {
                ops[op](a, b(ctx), { out: a });
              } else {
                ctx.i[token.v[symbolic]] = ops[op](a, b(ctx));
              }
            }
          }
        };
        while (check(t => t.t == TokenType.Bracket && (t.v == '(' || t.v == '['))) {
          const bracket = tokens.pop();
          if (bracket.v == '(') {
            // function call
            const token = expect();
            let call: ASTNode;
            if (token && token.t == TokenType.Bracket && token.v == ')') {
              tokens.pop();
              call = tryCall(tgt.g, []);
            } else {
              const args: ASTNode[] = [expr()];
              while (expect().t != TokenType.Bracket || cur().v != ')') {
                const sep = tokens.pop();
                if (sep.t != TokenType.Separator || sep.v != ',') {
                  throw new SyntaxError(`expected comma in argument list, got ${tokenTypes[sep.t]} ${sep.v}`);
                }
                args.push(expr());
              }
              tokens.pop();
              call = tryCall(tgt.g, args);
            }
            tgt = {
              g: call,
              s: () => {
                throw new TypeError('cannot assign to function call');
              },
              m: () => {
                throw new TypeError('cannot modify function call');
              }
            };
          } else if (bracket.v == '[') {
            // slice index
            let sliceParts: ASTNode[] = [];
            while (expect().t != TokenType.Bracket || cur().v != ']') {
              // hack for np.newaxis (+)
              while (
                (expect().t == TokenType.Separator && cur().v != ';') ||
                (cur().t == TokenType.Operator && (cur().v == '+' || cur().v == '...'))
              ) {
                sliceParts.push(() => tokens.pop().v);
              }
              if (expect().t == TokenType.Bracket && cur().v == ']') break;
              sliceParts.push(expr());
            }
            tokens.pop();
            const makeSlice = (ctx: Context) => sliceParts.map(p => p(ctx)).join('');
            const oldt = tgt;
            tgt = {
              g: (ctx: Context) => oldt.g(ctx)[makeSlice(ctx)],
              s: (v: ASTNode) => (ctx: Context) => {
                oldt.g(ctx)[makeSlice(ctx)] = v(ctx);
              },
              m: (op: string, b: ASTNode) => (ctx: Context) => {
                const base = oldt.g(ctx);
                const slice = makeSlice(ctx);
                if (base && base[ndvInternals]) {
                  const ar = base[slice];
                  ops[op](ar, b(ctx), { out: ar });
                } else {
                  base[slice] = ops[op](base[slice], b(ctx));
                }
              }
            };
          }
        }
        const oldresult = result;
        if (check(t => t.t == TokenType.Operator)) {
          const opMap = {
            '+=': 'add',
            '-=': 'sub',
            '*=': 'mul',
            '/=': 'div',
            '%=': 'mod',
            '@=': 'matmul',
            '**=': 'pow'
          };
          if (cur().v == '=') {
            tokens.pop();
            const set = tgt.s(expr());
            result = (ctx: Context) => {
              oldresult(ctx);
              set(ctx);
            };
          } else if (opMap[cur().v as string]) {
            tokens.pop();
            const modify = tgt.m(opMap[cur().v as string], expr());
            result = (ctx: Context) => {
              oldresult(ctx);
              modify(ctx);
            };
          }
        }
        if (result == oldresult) {
          tokens = prevTokens;
          const next = expr();
          let ret = check(t => t.t != TokenType.Separator || t.v != ';');
          result = (ctx: Context) => {
            oldresult(ctx);
            const val = next(ctx);
            if (ret) return val;
          }
          if (ret) break;
        }
      } else {
        const next = expr();
        let oldresult = result;
        let ret = check(t => t.t != TokenType.Separator || t.v != ';');
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
  
  const result = expr();
  if (tokens[0]) {
    throw new SyntaxError(`unexpected ${tokenTypes[tokens[0].t]} ${tokens[0].v}`);
  }

  return (args: Record<T, unknown>) => result({
    i: args,
    e: { ...ops }
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
  const parsed = parse(code, ...args.map((_, i) => i));
  const result = parsed(args);
  if (!result) {
    throw new TypeError('expression did not return a value');
  }
  return result;
}