use std::collections::BTreeMap;
use std::fmt::Write;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn object() -> Self {
        Value::Object(BTreeMap::new())
    }

    pub fn as_object_mut(&mut self) -> Option<&mut BTreeMap<String, Value>> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

pub fn to_string(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value);
    out
}

fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => write_number(out, *n),
        Value::String(s) => write_string(out, s),
        Value::Array(a) => {
            out.push('[');
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, v);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, v)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, k);
                out.push(':');
                write_value(out, v);
            }
            out.push('}');
        }
    }
}

fn write_number(out: &mut String, n: f64) {
    if n.is_finite() {
        if n == n.floor() && n.abs() < 1e16 {
            write!(out, "{}", n as i64).unwrap();
        } else {
            write!(out, "{}", n).unwrap();
        }
    } else {
        out.push_str("null");
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                write!(out, "\\u{:04x}", c as u32).unwrap();
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn to_vec(value: &Value) -> Vec<u8> {
    to_string(value).into_bytes()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub position: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "json parse error at {}: {}", self.position, self.message)
    }
}

impl std::error::Error for ParseError {}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

pub fn from_str(input: &str) -> Result<Value, ParseError> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(p.error("trailing characters"));
    }
    Ok(v)
}

impl<'a> Parser<'a> {
    fn error(&self, msg: &str) -> ParseError {
        ParseError {
            message: msg.to_string(),
            position: self.pos,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\n' | b'\r' | b'\t' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        self.skip_ws();
        if self.pos >= self.bytes.len() {
            return Err(self.error("unexpected end of input"));
        }
        match self.bytes[self.pos] {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => self.parse_string().map(Value::String),
            b't' | b'f' => self.parse_bool(),
            b'n' => self.parse_null(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            c => Err(ParseError {
                message: format!("unexpected character '{}'", c as char),
                position: self.pos,
            }),
        }
    }

    fn parse_object(&mut self) -> Result<Value, ParseError> {
        self.pos += 1;
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'}' {
            self.pos += 1;
            return Ok(Value::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.pos >= self.bytes.len() || self.bytes[self.pos] != b':' {
                return Err(self.error("expected ':'"));
            }
            self.pos += 1;
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_ws();
            if self.pos >= self.bytes.len() {
                return Err(self.error("unterminated object"));
            }
            match self.bytes[self.pos] {
                b',' => {
                    self.pos += 1;
                }
                b'}' => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.error("expected ',' or '}'")),
            }
        }
        Ok(Value::Object(map))
    }

    fn parse_array(&mut self) -> Result<Value, ParseError> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b']' {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            let value = self.parse_value()?;
            items.push(value);
            self.skip_ws();
            if self.pos >= self.bytes.len() {
                return Err(self.error("unterminated array"));
            }
            match self.bytes[self.pos] {
                b',' => {
                    self.pos += 1;
                }
                b']' => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.error("expected ',' or ']'")),
            }
        }
        Ok(Value::Array(items))
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        if self.pos >= self.bytes.len() || self.bytes[self.pos] != b'"' {
            return Err(self.error("expected '\"'"));
        }
        self.pos += 1;
        let mut s = String::new();
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            if c == b'"' {
                self.pos += 1;
                return Ok(s);
            }
            if c == b'\\' {
                self.pos += 1;
                if self.pos >= self.bytes.len() {
                    return Err(self.error("unterminated escape"));
                }
                match self.bytes[self.pos] {
                    b'"' => s.push('"'),
                    b'\\' => s.push('\\'),
                    b'/' => s.push('/'),
                    b'n' => s.push('\n'),
                    b'r' => s.push('\r'),
                    b't' => s.push('\t'),
                    b'b' => s.push('\u{08}'),
                    b'f' => s.push('\u{0c}'),
                    b'u' => {
                        if self.pos + 4 >= self.bytes.len() {
                            return Err(self.error("bad unicode escape"));
                        }
                        let hex = std::str::from_utf8(&self.bytes[self.pos + 1..self.pos + 5])
                            .map_err(|_| self.error("bad unicode escape"))?;
                        let code = u32::from_str_radix(hex, 16)
                            .map_err(|_| self.error("bad unicode escape"))?;
                        if let Some(ch) = char::from_u32(code) {
                            s.push(ch);
                        }
                        self.pos += 4;
                    }
                    _ => return Err(self.error("bad escape")),
                }
                self.pos += 1;
            } else {
                s.push(c as char);
                self.pos += 1;
            }
        }
        Err(self.error("unterminated string"))
    }

    fn parse_bool(&mut self) -> Result<Value, ParseError> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Value::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Value::Bool(false))
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn parse_null(&mut self) -> Result<Value, ParseError> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Value::Null)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn parse_number(&mut self) -> Result<Value, ParseError> {
        let start = self.pos;
        if self.bytes[self.pos] == b'-' {
            self.pos += 1;
        }
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            if c.is_ascii_digit()
                || c == b'.'
                || c == b'e'
                || c == b'E'
                || c == b'+'
                || c == b'-'
            {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.error("invalid number"))?;
        let n: f64 = s.parse().map_err(|_| self.error("invalid number"))?;
        Ok(Value::Number(n))
    }
}

pub trait ToJson {
    fn to_json(&self) -> Value;
}

pub trait FromJson: Sized {
    fn from_json(value: &Value) -> Result<Self, ParseError>;
}

impl ToJson for String {
    fn to_json(&self) -> Value {
        Value::String(self.clone())
    }
}

impl FromJson for String {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        match value {
            Value::String(s) => Ok(s.clone()),
            _ => Err(ParseError {
                message: "expected string".into(),
                position: 0,
            }),
        }
    }
}

impl ToJson for f64 {
    fn to_json(&self) -> Value {
        Value::Number(*self)
    }
}

impl FromJson for f64 {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        value
            .as_f64()
            .ok_or(ParseError { message: "expected number".into(), position: 0 })
    }
}

impl ToJson for bool {
    fn to_json(&self) -> Value {
        Value::Bool(*self)
    }
}

impl FromJson for bool {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        value
            .as_bool()
            .ok_or(ParseError { message: "expected bool".into(), position: 0 })
    }
}

impl<T: ToJson> ToJson for Vec<T> {
    fn to_json(&self) -> Value {
        Value::Array(self.iter().map(T::to_json).collect())
    }
}

impl<T: FromJson> FromJson for Vec<T> {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        match value {
            Value::Array(items) => items.iter().map(T::from_json).collect(),
            _ => Err(ParseError { message: "expected array".into(), position: 0 }),
        }
    }
}

impl<T: ToJson> ToJson for Option<T> {
    fn to_json(&self) -> Value {
        match self {
            Some(v) => v.to_json(),
            None => Value::Null,
        }
    }
}

impl<T: FromJson> FromJson for Option<T> {
    fn from_json(value: &Value) -> Result<Self, ParseError> {
        match value {
            Value::Null => Ok(None),
            other => T::from_json(other).map(Some),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_primitives() {
        for v in [
            Value::Null,
            Value::Bool(true),
            Value::Bool(false),
            Value::Number(3.5),
            Value::String("hi \"there\"\n".into()),
        ] {
            let s = to_string(&v);
            let parsed = from_str(&s).unwrap();
            assert_eq!(parsed, v);
        }
    }

    #[test]
    fn roundtrip_nested() {
        let mut obj = BTreeMap::new();
        obj.insert("a".into(), Value::Number(1.0));
        obj.insert("b".into(), Value::Array(vec![Value::Bool(true), Value::Null]));
        let v = Value::Object(obj);
        let s = to_string(&v);
        assert_eq!(from_str(&s).unwrap(), v);
    }

    #[test]
    fn parse_rejects_trailing() {
        assert!(from_str("1 2").is_err());
        assert!(from_str("{}}").is_err());
    }
}
