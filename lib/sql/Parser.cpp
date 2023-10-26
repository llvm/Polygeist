#include <string>
#include <vector>
#include <regex>
#include <algorithm>

#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"


#include "sql/SQLDialect.h"
#include "sql/SQLOps.h"
#include "sql/SQLTypes.h"


using namespace mlir;
using namespace sql;
enum class ParseType {
    Nothing = 0,
    Value = 1,
    Attribute = 2,
};

struct ParseValue {
private:
    ParseType ty;
    Value value;
    Attribute attr;
public:
    ParseValue() : ty(ParseType::Nothing), value(nullptr), attr(nullptr) {}
    ParseValue(Value value) : ty(ParseType::Value), value(value), attr(nullptr) {
        assert(value);
    }
    ParseValue(Attribute attr) : ty(ParseType::Attribute), value(nullptr), attr(attr) {}

    ParseType getType() const {
        return ty;
    }
    Value getValue() const {
        // llvm::errs() << "ty: " << type_value << "\n";
        // llvm::errs() << "value: " << value << "\n";
        // llvm::errs() << "attr: " << attr << "\n";
        assert(ty == ParseType::Value);
        assert(value);
        return value;
    }
    Attribute getAttr() const {
        assert(ty == ParseType::Attribute);
        return attr;
    }
};

enum class ParseMode {
    None,
    Column,
    Table,
    Bool, 
};


template<typename T>
std::ostream& operator<<(typename std::enable_if<std::is_enum<T>::value, std::ostream>::type& stream, const T& e)
{
    return stream << static_cast<typename std::underlying_type<T>::type>(e);
}

class SQLParser {

public:
    Location loc;
    OpBuilder &builder;
    std::string sql;
    unsigned int i;

    static std::vector<std::string> reservedWords; 

    
    SQLParser(Location loc, OpBuilder &builder, std::string sql, int i) : loc(loc), builder(builder),
        sql(sql), i(i) {
        }

    std::string peek() {
        auto [peeked, _] = peekWithLength();
        return peeked;
    }

    std::string pop() {
        auto [peeked, len] = peekWithLength();
        i += len;
        popWhitespace();
        return peeked;
    }

    void popWhitespace() {
        // it doesn't recognize 
        while (i < sql.size() && (sql[i] == ' ' || sql[i] == '\n' || sql[i] == '\t')) {
            i++;
        }
    }

    std::pair<std::string, int> peekWithLength() {
        if (i >= sql.size()) {
            return {"", 0};
        }
        for (std::string rWord : reservedWords) {
            auto token = sql.substr(i, std::min(sql.size()-i, rWord.size()));
            std::transform(token.begin(), token.end(), token.begin(), ::toupper);
            if (token == rWord) {
                return {token, static_cast<int>(token.size())};
            }
        }
        if (sql[i] == '\'') { // Quoted string
            return peekQuotedStringWithLength();
        }
        return peekIdentifierWithLength();
    }

    std::pair<std::string, int> peekQuotedStringWithLength() {
        if (sql.size() < i || sql[i] != '\'') {
            return {"", 0};
        }
        for (unsigned int j = i + 1; j < sql.size(); j++) {
            if (sql[j] == '\'' && sql[j-1] != '\\') {
                return {sql.substr(i + 1, j - (i + 1)), j - i + 2}; // +2 for the two quotes
            }
        }
        return {"", 0};
    }

    std::pair<std::string, int> peekIdentifierWithLength() {
        std::regex e("[a-zA-Z0-9_*]");
        for (unsigned int j = i; j < sql.size(); j++) {
            if (!std::regex_match(std::string(1, sql[j]), e)) {
                return {sql.substr(i, j - i), j - i};
            }
        }
        return {sql.substr(i), static_cast<int>(sql.size())-i};
    }


    bool is_number(std::string* s){
        std::string::const_iterator it = s->begin();
        while (it != s->end() && std::isdigit(*it)) ++it;
        return !s->empty() && it == s->end();
    }


    // Parse the next command, if any
    ParseValue parseNext(ParseMode mode) {
        // for (unsigned int j = i; j < sql.size(); j++) {
        //     auto peekStr = peek(); 
        //     pop(); 
        //     llvm::errs() << "peekStrTest: " << i << " " << peekStr << "\n";
        // }
        if (i >= sql.size()) {
            return ParseValue();
        }
        auto peekStr = peek();
        llvm::errs() << "peekStr: " << peekStr << "\n";
        assert(peekStr.size() > 0); 

        if (peekStr == "SELECT") {
            assert(mode == ParseMode::None || mode == ParseMode::Table);
            pop();
            peekStr = peek(); 
            if (peekStr == "DISTINCT") {
                pop();
                // do something different here
            }
            llvm::SmallVector<Value> columns;
            bool hasColumns = true;
            bool hasWhere = false; 
            bool selectAll = false; 
            int limit = -1;
            Value table = nullptr;
            while (true) {
                peekStr = peek();
                if (hasColumns) { 
                    if (peekStr == "FROM") {
                        pop();
                        table = parseNext(ParseMode::Table).getValue();
                        llvm::errs() << "table: " << table << "\n";
                        hasColumns = false;
                        break;
                    }
                    if (peekStr == "*") {
                        pop();
                        selectAll = true; 
                        continue;
                    }
                    if (peekStr == ",") {
                        pop();
                        llvm::errs() << "comma\n";
                        continue;
                    }
                    ParseValue col = parseNext(ParseMode::Column);
                    if (col.getType() == ParseType::Nothing) {
                        hasColumns = false; 
                        break;
                    } else {
                        columns.push_back(col.getValue());
                    }
                    
                } else if (peekStr == "WHERE") {
                    pop(); 
                    hasWhere = true; 
                } else if (peekStr == "LIMIT"){
                    pop(); 
                    peekStr = peek();
                    if (peekStr == "ALL"){
                        pop();
                    } else if (is_number(&peekStr)){
                        pop();
                        limit = std::stoi(peekStr);
                    }
                } else { 
                    // break;
                    assert(0 && " additional clauses like where/etc not yet handled");
                }
            }
            if (!table){
                llvm::errs() << " table is null: " << table << "\n";
                table = builder.create<sql::TableOp>(loc, ExprType::get(builder.getContext()), builder.getStringAttr("")).getResult();
            }
            // if (selectAll){
            //     assert(table && "table cannot be null");
            //     return ParseValue(builder.create<sql::SelectAllOp>(loc, ExprType::get(builder.getContext()), table).getResult());
            // } else {
            return ParseValue(builder.create<sql::SelectOp>(loc, ExprType::get(builder.getContext()), columns, table, selectALl, limit).getResult());
            // }
        } else if (is_number(&peekStr)){
            pop();
            return ParseValue(builder.create<IntOp>(loc, ExprType::get(builder.getContext()), builder.getStringAttr(peekStr)).getResult());
        } else if (mode == ParseMode::Column) {
            // do we need this?? 
            if (peekStr == "*") {
                pop();
                return ParseValue(builder.create<AllColumnsOp>(loc, ExprType::get(builder.getContext()), builder.getStringAttr(peekStr)).getResult());
            }
            pop();
            return ParseValue(builder.create<ColumnOp>(loc, ExprType::get(builder.getContext()),  builder.getStringAttr(peekStr)).getResult());
        } else if (mode == ParseMode::Table) {
            pop();
            return ParseValue(builder.create<TableOp>(loc,ExprType::get(builder.getContext()), builder.getStringAttr(peekStr)).getResult());
        } else if (mode == ParseMode::Bool) {
            // col = peekStr;
            pop();
           
        } else if (peekStr == "(") {
            pop();
            ParseValue res = parseNext(ParseMode::None);
            assert(peek() == ")");
            pop();
            return res;
        } else if (peekStr == ")") {
            return ParseValue();
        } 
        llvm::errs() << " Unknown token to parse: " << peekStr << "\n";
        llvm_unreachable("Unknown token to parse");
    }

};

std::vector<std::string> SQLParser::reservedWords = {
    "(", ")", ">=", "<=", "!=", ",", "=", ">", "<", ",", "SELECT", "DISTINCT", "INSERT INTO", "VALUES", "UPDATE", "DELETE FROM", "WHERE", "FROM", "SET", "AS"
};


mlir::Value parseSQL(mlir::Location loc, mlir::OpBuilder& builder, std::string str) {
    SQLParser parser(loc, builder, std::string(str), 0);
    auto resOp = parser.parseNext(ParseMode::None);
    
    return resOp.getValue();
}