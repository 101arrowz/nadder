import { ndvInternals } from '../../util';
import * as ops from '../../util/ufunc/ops';

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

export function evaluate(strings: TemplateStringsArray, ...args: unknown[]) {
  if (!strings.length || args.length != strings.length - 1) {
    throw new TypeError('invalid arguments to evaluate');
  }
  let curStringInd = 0;
  let curInput = strings[0];
  const tokens: Token[] = [];
  o: do {
    curInput = curInput.trimStart();
    while (!curInput) {
      if (++curStringInd >= strings.length) break o;
      tokens.push({ t: TokenType.Value, v: args[curStringInd - 1] });
      curInput = strings[curStringInd].trimStart();
    }
    let char = curInput[0];
    switch (char) {
      case ',':
      case ':':
        tokens.push({ t: TokenType.Separator, v: char });
        curInput = curInput.slice(1);
        break;
      case '+':
      case '-':
      case '/':
      case '%':
        tokens.push({ t: TokenType.Operator, v: char });
        curInput = curInput.slice(1);
        break;
      case '*':
        let result = curInput[1] == '*' ? '**' : '*';
        curInput = curInput.slice(result.length);
        tokens.push({ t: TokenType.Operator, v: result });
        break;
      case '(':
      case ')':
      case '[':
      case ']':
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
        // 1-9
        if (code > 47 && code < 58) {
          let num = char;
          let ind = 1;
          while (ind < curInput.length) {
            const newCode = curInput.charCodeAt(ind);
            // 0-9
            if (newCode < 48 || newCode > 57) break;
            num += curInput[ind++];
          }
          if (num.length > 1 && code == 48) {
            throw new TypeError('leading zeros are not allowed in integer literals');
          }
          tokens.push({ t: TokenType.Value, v: +num });
          curInput = curInput.slice(ind);
          break;
        }
        // a-z (no capitals)
        if (code > 96 && code < 123) {
          let ident = char;
          let ind = 1;
          while (ind < curInput.length) {
            const newCode = curInput.charCodeAt(ind);
            // a-z0-9  
            if ((newCode < 97 || newCode > 122) && (newCode < 65 || newCode > 90)) break;
            ident += curInput[ind++];
          }
          tokens.push({ t: TokenType.Identifier, v: ident });
          curInput = curInput.slice(ind);
          break;
        }
        throw new SyntaxError(`could not parse token from ${char}`);
    }
  } while (true)
  
  const context = { ...ops };

  const cur = () => {
    if (tokens[0]) return tokens[0];
    throw new SyntaxError('unexpected EOF in expression');
  }

  const tryCall = (fn: unknown, args: unknown[]) => {
    if (typeof fn != 'function') {
      throw new TypeError(`attempted to call non-function ${fn}`);
    }
    return fn(...args);
  }

  const maybeBracket = (val: unknown) => {
    const bracket = tokens[0];
    if (bracket && bracket.t == TokenType.Bracket && (bracket.v == '(' || bracket.v == '[')) {
      tokens.shift();
      if (bracket.v == '(') {
        // function call
        const token = cur();
        if (token.t == TokenType.Bracket && token.v == ')') {
          tokens.shift();
          return tryCall(val, []);
        }
        const args: unknown[] = [expr()];
        while (cur().t != TokenType.Bracket || cur().v != ')') {
          const sep = tokens.shift();
          if (sep.t != TokenType.Separator || sep.v != ',') {
            throw new SyntaxError(`expected comma after ${args[args.length - 1]}`);
          }
          args.push(expr());   
        }
        tokens.shift();
        return tryCall(val, args);
      } else if (bracket.v == '[') {
        // slice index
        if (!val || !val[ndvInternals]) throw new TypeError(`attempted to index non-ndarray ${val}`);
        let sliceStr = '';
        while (cur().t != TokenType.Bracket || cur().v != ']') {
          // hack for np.newaxis (+)
          while (
            cur().t == TokenType.Separator ||
            (cur().t == TokenType.Operator && (cur().v == '+' || cur().v == '...'))
          ) {
            sliceStr += tokens.shift().v;
          }
          if (cur().t == TokenType.Bracket && cur().v == ']') break;
          sliceStr += expr();
        }
        tokens.shift();
        return val[sliceStr];
      }
    }
    return val;
  };

  const unit = () => {
    const token = cur();
    if (token.t == TokenType.Value) {
      tokens.shift();
      return maybeBracket(token.v);
    }
    if (token.t == TokenType.Operator) {
      tokens.shift();
      if (token.v == '-') return ops.neg(unit());
      if (token.v == '+') return ops.pos(unit());
      throw new SyntaxError(`could not parse operator ${token.v}`);
    }
    if (token.t == TokenType.Identifier) {
      const name = token.v;
      tokens.shift();
      if (context[name]) return maybeBracket(context[name]);
      throw new ReferenceError(`unknown identifier ${name}`);
    }
    if (token.t == TokenType.Bracket) {
      let result: unknown;
      if (token.v == '(') {
        tokens.shift();
        result = expr();
        if (cur().t != TokenType.Bracket || cur().v != ')') {
          throw new SyntaxError('expected closing bracket');
        }
        tokens.shift();
      } else if (token.v == '[') {
        tokens.shift();
        if (cur().t == TokenType.Bracket && cur().v == ']') {
          tokens.shift();
          result = [];
        } else {
          result = [expr()];
          while (cur().t != TokenType.Bracket || cur().v != ']') {
            const sep = tokens.shift();
            if (sep.t != TokenType.Separator || sep.v != ',') {
              throw new SyntaxError(`expected comma after ${(result as unknown[])[(result as unknown[]).length - 1]}`);
            }
            (result as unknown[]).push(expr());
          }
          tokens.shift();
        }
      }
      return maybeBracket(result);
    }
    throw new SyntaxError(`could not parse expression`);
  };

  const powExpr = () => {
    let left = unit();
    while (tokens[0] && tokens[0].t == TokenType.Operator) {
      switch (tokens[0].v) {
        case '**':
          tokens.shift();
          const right = unit();
          if (left === Math.E) left = ops.exp(right);
          else if (left === 2) left = ops.exp2(right);
          else left = ops.pow(left, right);
          break;
        default: return left;
      }
    }
    return left;
  };

  const mulExpr = () => {
    let left = powExpr();
    while (tokens[0] && tokens[0].t == TokenType.Operator) {
      switch (tokens[0].v) {
        case '*':
          tokens.shift();
          left = ops.mul(left, powExpr());
          break;
        case '/':
          tokens.shift();
          left = ops.div(left, powExpr());
          break;
        case '%':
          tokens.shift();
          left = ops.mod(left, powExpr());
          break;
        default: return left;
      }
    }
    return left;
  };

  const addExpr = () => {
    let left = mulExpr();
    while (tokens[0] && tokens[0].t == TokenType.Operator) {
      switch (tokens[0].v) {
        case '+':
          tokens.shift();
          left = ops.add(left, mulExpr());
          break;
        case '-':
          tokens.shift();
          left = ops.sub(left, mulExpr());
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
  }
  const result = expr();
  if (tokens[0]) {
    throw new SyntaxError(`could not parse expression`);
  }
  return result;
}